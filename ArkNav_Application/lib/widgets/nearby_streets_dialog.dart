import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

// ============================================================================
// DATA MODELS
// ============================================================================

class FloodForecast {
  final String locationName;
  final double lat;
  final double lon;
  final FloodRiskData currentRisk;
  final SevenDayForecast sevenDayForecast;

  FloodForecast({
    required this.locationName,
    required this.lat,
    required this.lon,
    required this.currentRisk,
    required this.sevenDayForecast,
  });

  factory FloodForecast.fromJson(Map<String, dynamic> json) {
    return FloodForecast(
      locationName: json['locationName'] ?? 'Unknown',
      lat: json['current_flood_risk']['input_coordinates']['lat'],
      lon: json['current_flood_risk']['input_coordinates']['lon'],
      currentRisk: FloodRiskData.fromJson(json['current_flood_risk']),
      sevenDayForecast: SevenDayForecast.fromJson(
        json['seven_days_flood_risk'],
      ),
    );
  }
}

class FloodRiskData {
  final String category;
  final double predictedHeight;
  final bool inZone;

  FloodRiskData({
    required this.category,
    required this.predictedHeight,
    required this.inZone,
  });

  factory FloodRiskData.fromJson(Map<String, dynamic> json) {
    return FloodRiskData(
      category: json['category'] ?? 'Unknown',
      predictedHeight: json['predicted_flood_height_m'] ?? 0.0,
      inZone: json['features']['in_zone_int'] == 1,
    );
  }
}

class SevenDayForecast {
  final List<DailyForecast> forecasts;

  SevenDayForecast({required this.forecasts});

  factory SevenDayForecast.fromJson(Map<String, dynamic> json) {
    final forecastList = json['seven_day_forecast'] as List;
    return SevenDayForecast(
      forecasts: forecastList.map((f) => DailyForecast.fromJson(f)).toList(),
    );
  }
}

class DailyForecast {
  final String date;
  final String category;
  final double predictedHeight;
  final double precipitation;
  final bool inZone;

  DailyForecast({
    required this.date,
    required this.category,
    required this.predictedHeight,
    required this.precipitation,
    required this.inZone,
  });

  factory DailyForecast.fromJson(Map<String, dynamic> json) {
    return DailyForecast(
      date: json['date'] ?? '',
      category: json['category'] ?? 'Unknown',
      predictedHeight: json['predicted_flood_height_m'] ?? 0.0,
      precipitation: json['precipitation_mm'] ?? 0.0,
      inZone: json['features']['in_zone_int'] == 1,
    );
  }

  // Get color based on category
  Color getCategoryColor() {
    switch (category) {
      case 'No Flood':
        return Colors.green;
      case 'Ankle High':
        return Colors.lightGreen;
      case 'Knee High':
        return Colors.yellow;
      case 'Waist High':
        return Colors.orange;
      case 'Neck High':
        return Colors.deepOrange;
      case 'Higher than Neck':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }
}

// ============================================================================
// API SERVICE
// ============================================================================

class FloodPredictionService {
  static const String baseUrl =
      'http://10.0.2.2:5000'; // Replace with actual URL

  /// Fetch flood predictions for multiple locations
  static Future<List<FloodForecast>> fetchFloodForecasts(
    List<Map<String, dynamic>> locations,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl/predict_flood_risk'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(locations),
          )
          .timeout(const Duration(seconds: 90));

      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        return data.map((json) => FloodForecast.fromJson(json)).toList();
      } else {
        throw Exception('Failed to fetch flood predictions');
      }
    } catch (e) {
      print('Error fetching flood forecasts: $e');
      rethrow;
    }
  }

  /// Fetch nearby streets and their coordinates
  static Future<List<Map<String, dynamic>>> fetchNearbyStreetsWithCoords(
    double lat,
    double lon,
    int radius,
  ) async {
    final overpassQuery = """
    [out:json][timeout:15];
    (
      way(around:$radius,$lat,$lon)["highway"];
    );
    out center;
  """;

    try {
      print(
        'Fetching streets around: $lat, $lon with radius: $radius',
      ); // DEBUG

      final response = await http
          .post(
            Uri.parse('https://overpass-api.de/api/interpreter'),
            body: overpassQuery,
          )
          .timeout(const Duration(seconds: 15));

      print('Overpass response status: ${response.statusCode}'); // DEBUG

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        List<Map<String, dynamic>> streets = [];

        print('Overpass elements: ${data['elements']?.length ?? 0}'); // DEBUG

        for (var element in data['elements']) {
          if (element['tags'] != null &&
              element['tags']['name'] != null &&
              element['center'] != null) {
            streets.add({
              'name': element['tags']['name'],
              'lat': element['center']['lat'],
              'lon': element['center']['lon'],
            });
            print('Found street: ${element['tags']['name']}'); // DEBUG
          }
        }

        print('Total streets with names: ${streets.length}'); // DEBUG
        return streets;
      }
    } catch (e) {
      print('Error fetching streets: $e');
    }

    return [];
  }
}

// ============================================================================
// ENHANCED DIALOG
// ============================================================================

Future<void> showFloodForecastDialog(
  BuildContext context,
  LatLng latLng, {
  required String locationLabel,
}) async {
  showDialog(
    context: context,
    barrierDismissible: false,
    builder: (BuildContext context) {
      return Dialog(
        backgroundColor: const Color(0xFF1D364E),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        child: FutureBuilder<Map<String, dynamic>>(
          future: _loadFloodData(latLng, locationLabel),
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return _buildLoadingView();
            } else if (snapshot.hasError) {
              return _buildErrorView(context, snapshot.error.toString());
            } else if (!snapshot.hasData) {
              return _buildErrorView(context, 'No data available');
            }

            final data = snapshot.data!;
            final mainForecast = data['mainForecast'] as FloodForecast;
            // FIX: Remove the type cast, it's already the correct type
            final streetForecasts = data['streetForecasts'] as List;
            final streetForecastList = streetForecasts.cast<FloodForecast>();

            return _buildForecastView(
              context,
              mainForecast,
              streetForecastList,
              locationLabel,
            );
          },
        ),
      );
    },
  );
}

/// Load all flood data (main location + nearby streets)
Future<Map<String, dynamic>> _loadFloodData(
  LatLng latLng,
  String locationLabel,
) async {
  // Fetch nearby streets with coordinates
  final streets = await FloodPredictionService.fetchNearbyStreetsWithCoords(
    latLng.latitude,
    latLng.longitude,
    300,
  );

  print('Found ${streets.length} nearby streets');

  // Remove duplicate street names, keep only unique streets
  final uniqueStreets = <String, Map<String, dynamic>>{};
  for (var street in streets) {
    final name = street['name'] as String;
    if (!uniqueStreets.containsKey(name)) {
      uniqueStreets[name] = street;
    }
  }

  final uniqueStreetsList = uniqueStreets.values.toList();
  print('Unique streets: ${uniqueStreetsList.length}');

  // Optionally limit to first 15 unique streets
  final limitedStreets = uniqueStreetsList.take(3).toList();
  print('Using ${limitedStreets.length} streets for forecast');

  // Build request payload
  final locations = [
    {
      'locationName': locationLabel,
      'lat': latLng.latitude,
      'lon': latLng.longitude,
    },
    ...limitedStreets.map(
      (street) => {
        'locationName': street['name'],
        'lat': street['lat'],
        'lon': street['lon'],
      },
    ),
  ];

  print('Requesting forecasts for ${locations.length} locations');

  // Fetch flood predictions for all locations
  final forecasts = await FloodPredictionService.fetchFloodForecasts(locations);

  print('Received ${forecasts.length} forecasts');

  final List<FloodForecast> streetForecastsList =
      forecasts.length > 1 ? forecasts.sublist(1) : [];

  print('Street forecasts: ${streetForecastsList.length}');

  return {
    'mainForecast': forecasts.first,
    'streetForecasts': streetForecastsList,
  };
}

// ============================================================================
// UI COMPONENTS
// ============================================================================

Widget _buildLoadingView() {
  return Container(
    padding: const EdgeInsets.all(40),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        CircularProgressIndicator(color: Color(0xFF01AFBA)),
        SizedBox(height: 20),
        Text(
          'Loading flood forecasts...',
          style: TextStyle(color: Colors.white70),
        ),
      ],
    ),
  );
}

Widget _buildErrorView(BuildContext context, String error) {
  return Container(
    padding: const EdgeInsets.all(24),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(Icons.error_outline, color: Colors.red, size: 48),
        const SizedBox(height: 16),
        Text(
          'Failed to load flood data',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          error,
          style: const TextStyle(color: Colors.white70),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 20),
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text(
            'Close',
            style: TextStyle(color: Color(0xFF01AFBA)),
          ),
        ),
      ],
    ),
  );
}

Widget _buildForecastView(
  BuildContext context,
  FloodForecast mainForecast,
  List<FloodForecast> streetForecasts,
  String locationLabel,
) {
  return Container(
    constraints: const BoxConstraints(maxHeight: 600, maxWidth: 500),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Header
        _buildHeader(context, locationLabel),

        // Main location forecast
        Expanded(
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildMainLocationCard(mainForecast),

                const Padding(
                  padding: EdgeInsets.all(16),
                  child: Divider(color: Colors.white24, thickness: 1),
                ),

                // 7-day forecast for main location
                _buildSevenDayForecast(mainForecast),

                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Divider(color: Colors.white24, thickness: 1),
                ),

                // Nearby streets section
                _buildNearbyStreetsSection(streetForecasts),
              ],
            ),
          ),
        ),

        // Close button
        Padding(
          padding: const EdgeInsets.all(16),
          child: SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.pop(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF01AFBA),
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text(
                'Close',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
      ],
    ),
  );
}

Widget _buildHeader(BuildContext context, String locationLabel) {
  return Container(
    padding: const EdgeInsets.all(20),
    decoration: const BoxDecoration(
      color: Color(0xFF2A4A6F),
      borderRadius: BorderRadius.only(
        topLeft: Radius.circular(20),
        topRight: Radius.circular(20),
      ),
    ),
    child: Row(
      children: [
        const Icon(Icons.location_on, color: Color(0xFF01AFBA), size: 28),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Flood Forecast',
                style: TextStyle(
                  color: Color(0xFF01AFBA),
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                locationLabel,
                style: const TextStyle(color: Colors.white70, fontSize: 14),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

Widget _buildMainLocationCard(FloodForecast forecast) {
  return Container(
    margin: const EdgeInsets.all(16),
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: const Color(0xFF2A4A6F),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: const Color(0xFF01AFBA), width: 2),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color:
                    forecast.currentRisk.inZone
                        ? Colors.red.withOpacity(0.3)
                        : Colors.green.withOpacity(0.3),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                forecast.currentRisk.inZone ? 'Flood Zone' : 'Safe Zone',
                style: TextStyle(
                  color:
                      forecast.currentRisk.inZone ? Colors.red : Colors.green,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ),
            const Spacer(),
            const Text(
              'Current Risk',
              style: TextStyle(color: Colors.white70, fontSize: 12),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Container(
              width: 8,
              height: 60,
              decoration: BoxDecoration(
                color: _getCategoryColor(forecast.currentRisk.category),
                borderRadius: BorderRadius.circular(4),
              ),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  forecast.currentRisk.category,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${forecast.currentRisk.predictedHeight.toStringAsFixed(2)}m',
                  style: const TextStyle(
                    color: Color(0xFF01AFBA),
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ],
        ),
      ],
    ),
  );
}

Widget _buildSevenDayForecast(FloodForecast forecast) {
  return Padding(
    padding: const EdgeInsets.symmetric(horizontal: 16),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '7-Day Forecast',
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 12),
        ...forecast.sevenDayForecast.forecasts.map((daily) {
          return _buildDailyForecastCard(daily);
        }).toList(),
      ],
    ),
  );
}

Widget _buildDailyForecastCard(DailyForecast daily) {
  return Container(
    margin: const EdgeInsets.only(bottom: 8),
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: const Color(0xFF2A4A6F).withOpacity(0.5),
      borderRadius: BorderRadius.circular(12),
    ),
    child: Row(
      children: [
        // Date
        SizedBox(
          width: 70,
          child: Text(
            _formatDate(daily.date),
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
        ),
        const SizedBox(width: 8),

        // Category indicator
        Container(
          width: 4,
          height: 40,
          decoration: BoxDecoration(
            color: daily.getCategoryColor(),
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 12),

        // Details
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                daily.category,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                '${daily.predictedHeight.toStringAsFixed(2)}m',
                style: const TextStyle(color: Color(0xFF01AFBA), fontSize: 12),
              ),
            ],
          ),
        ),

        // Precipitation
        Row(
          children: [
            const Icon(
              Icons.water_drop,
              color: Colors.lightBlueAccent,
              size: 14,
            ),
            const SizedBox(width: 4),
            Text(
              '${daily.precipitation.toStringAsFixed(1)}mm',
              style: const TextStyle(color: Colors.white70, fontSize: 12),
            ),
          ],
        ),
      ],
    ),
  );
}

Widget _buildNearbyStreetsSection(List<FloodForecast> streetForecasts) {
  if (streetForecasts.isEmpty) {
    return const Padding(
      padding: EdgeInsets.all(16),
      child: Text(
        'No nearby streets found',
        style: TextStyle(color: Colors.white70),
      ),
    );
  }

  return Padding(
    padding: const EdgeInsets.symmetric(horizontal: 16),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Nearby Streets',
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 12),
        ...streetForecasts.map((street) {
          return _buildStreetCard(street);
        }).toList(),
        const SizedBox(height: 16),
      ],
    ),
  );
}

Widget _buildStreetCard(FloodForecast street) {
  return Container(
    margin: const EdgeInsets.only(bottom: 12), // ADD THIS LINE
    child: ExpansionTile(
      tilePadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      backgroundColor: const Color(0xFF2A4A6F).withOpacity(0.3),
      collapsedBackgroundColor: const Color(0xFF2A4A6F).withOpacity(0.3),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      collapsedShape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      title: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: street.currentRisk.inZone ? Colors.red : Colors.green,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              street.locationName,
              style: const TextStyle(color: Colors.white, fontSize: 14),
            ),
          ),
        ],
      ),
      subtitle: Padding(
        padding: const EdgeInsets.only(top: 4, left: 16),
        child: Text(
          '${street.currentRisk.category} - ${street.currentRisk.predictedHeight.toStringAsFixed(2)}m',
          style: TextStyle(
            color: _getCategoryColor(street.currentRisk.category),
            fontSize: 12,
          ),
        ),
      ),
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          child: Column(
            children:
                street.sevenDayForecast.forecasts.map((daily) {
                  return _buildCompactDailyForecast(daily);
                }).toList(),
          ),
        ),
      ],
    ),
  );
}

Widget _buildCompactDailyForecast(DailyForecast daily) {
  return Container(
    margin: const EdgeInsets.only(bottom: 6),
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
    decoration: BoxDecoration(
      color: Colors.black26,
      borderRadius: BorderRadius.circular(8),
    ),
    child: Row(
      children: [
        SizedBox(
          width: 50,
          child: Text(
            _formatDate(daily.date),
            style: const TextStyle(color: Colors.white70, fontSize: 11),
          ),
        ),
        Container(
          width: 3,
          height: 24,
          margin: const EdgeInsets.symmetric(horizontal: 8),
          decoration: BoxDecoration(
            color: daily.getCategoryColor(),
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        Expanded(
          child: Text(
            daily.category,
            style: const TextStyle(color: Colors.white, fontSize: 11),
          ),
        ),
        Text(
          '${daily.predictedHeight.toStringAsFixed(1)}m',
          style: const TextStyle(
            color: Color(0xFF01AFBA),
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(width: 8),
        Icon(
          Icons.water_drop,
          color: Colors.lightBlueAccent.withOpacity(0.7),
          size: 10,
        ),
        Text(
          ' ${daily.precipitation.toStringAsFixed(0)}',
          style: const TextStyle(color: Colors.white70, fontSize: 10),
        ),
      ],
    ),
  );
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

Color _getCategoryColor(String category) {
  switch (category) {
    case 'No Flood':
      return Colors.green;
    case 'Ankle High':
      return Colors.lightGreen;
    case 'Knee High':
      return Colors.yellow;
    case 'Waist High':
      return Colors.orange;
    case 'Neck High':
      return Colors.deepOrange;
    case 'Higher than Neck':
      return Colors.red;
    default:
      return Colors.grey;
  }
}

String _formatDate(String date) {
  try {
    final DateTime dt = DateTime.parse(date);
    return '${dt.month}/${dt.day}';
  } catch (e) {
    return date;
  }
}
