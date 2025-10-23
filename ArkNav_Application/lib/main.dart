import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'screens/settings.dart';
import 'widgets/nearby_streets_dialog.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MapWithUI(),
    );
  }
}

class MapWithUI extends StatefulWidget {
  const MapWithUI({super.key});

  @override
  State<MapWithUI> createState() => _MapWithUIState();
}

class _MapWithUIState extends State<MapWithUI> {
  LatLng? _currentLatLng;
  LatLng? _pinnedLatLng;
  final TextEditingController _locationController = TextEditingController();
  final TextEditingController _pinLocationController = TextEditingController();
  String selectedDuration = 'Now';
  final MapController _mapController = MapController();

  // Optimization: Cache location stream and debounce updates
  StreamSubscription<Position>? _positionStreamSubscription;
  Timer? _debounceTimer;
  bool _isLoadingLocation = false;
  String? _cachedAddress;
  LatLng? _lastGeocodedPosition;
  List<dynamic> suggestions = [];

  @override
  void initState() {
    super.initState();
    _initializeLocationTracking();
  }

  @override
  void dispose() {
    _positionStreamSubscription?.cancel();
    _debounceTimer?.cancel();
    _locationController.dispose();
    _pinLocationController.dispose();
    super.dispose();
  }

  Future<List<dynamic>> fetchLocationSuggestions(String input) async {
    if (input.isEmpty) return [];

    final url = Uri.parse(
      'https://nominatim.openstreetmap.org/search'
      '?q=$input'
      '&format=json'
      '&addressdetails=1'
      '&limit=5'
      '&countrycodes=ph', // 🔹 Restrict to Philippines only
    );

    final response = await http.get(
      url,
      headers: {
        'User-Agent':
            'ArkNav/1.0 (arknavadmin@arknav.com)', // Nominatim requires this
      },
    );

    if (response.statusCode == 200) {
      final List data = json.decode(response.body);
      return data;
    } else {
      return [];
    }
  }

  // === OPTIMIZED LOCATION TRACKING ===

  /// Initialize continuous location tracking with position stream
  void _initializeLocationTracking() async {
    try {
      // Check permissions once
      final hasPermission = await _checkLocationPermission();
      if (!hasPermission) return;

      // Get initial position immediately
      final initialPosition = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      await _updateLocationUI(initialPosition);

      // Start listening to location updates (more efficient than manual polling)
      _positionStreamSubscription = Geolocator.getPositionStream(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          distanceFilter: 10, // Only update if moved 10+ meters
        ),
      ).listen(
        (Position position) {
          _updateLocationUI(position);
        },
        onError: (error) {
          print("Location stream error: $error");
        },
      );
    } catch (e) {
      print("ERROR initializing location: $e");
      _showErrorSnackBar("Failed to get location. Please enable GPS.");
    }
  }

  /// Check and request location permissions
  Future<bool> _checkLocationPermission() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      _showErrorSnackBar('Location services are disabled.');
      return false;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        _showErrorSnackBar('Location permissions denied.');
        return false;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      _showErrorSnackBar('Location permissions permanently denied.');
      return false;
    }

    return true;
  }

  /// Update UI with new position (with caching to avoid unnecessary geocoding)
  Future<void> _updateLocationUI(Position position) async {
    final newLatLng = LatLng(position.latitude, position.longitude);

    // Optimization: Only geocode if moved significantly (>50m from last geocoded position)
    final shouldGeocode =
        _lastGeocodedPosition == null ||
        _calculateDistance(_lastGeocodedPosition!, newLatLng) > 0.05; // ~50m

    setState(() {
      _currentLatLng = newLatLng;
    });

    if (shouldGeocode && !_isLoadingLocation) {
      _isLoadingLocation = true;
      _lastGeocodedPosition = newLatLng;

      // Debounce geocoding requests (wait 2 seconds after last update)
      _debounceTimer?.cancel();
      _debounceTimer = Timer(const Duration(seconds: 2), () async {
        try {
          final address = await _getAddressFromPosition(position);
          if (mounted) {
            setState(() {
              _cachedAddress = address;
              _locationController.text = address;
              _isLoadingLocation = false;
            });
          }
        } catch (e) {
          print("Geocoding error: $e");
          _isLoadingLocation = false;
        }
      });
    } else if (_cachedAddress != null) {
      // Use cached address if available
      _locationController.text = _cachedAddress!;
    }
  }

  /// Calculate distance between two points in degrees (approximate)
  double _calculateDistance(LatLng point1, LatLng point2) {
    return ((point1.latitude - point2.latitude).abs() +
        (point1.longitude - point2.longitude).abs());
  }

  /// Get address from position (with error handling)
  Future<String> _getAddressFromPosition(Position position) async {
    try {
      List<Placemark> placemarks = await placemarkFromCoordinates(
        position.latitude,
        position.longitude,
      ).timeout(const Duration(seconds: 5));

      if (placemarks.isNotEmpty) {
        final place = placemarks.first;
        return '${place.street ?? ''}, ${place.locality ?? ''}, ${place.country ?? ''}'
            .replaceAll(RegExp(r'^, |, $'), ''); // Clean up empty parts
      }
    } catch (e) {
      print("Geocoding timeout or error: $e");
    }
    return 'Lat: ${position.latitude.toStringAsFixed(4)}, Lon: ${position.longitude.toStringAsFixed(4)}';
  }

  /// Manual refresh location (for refresh button)
  void _refreshCurrentLocation() async {
    try {
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      await _updateLocationUI(position);

      // Move map to current location
      if (_currentLatLng != null) {
        _mapController.move(_currentLatLng!, 15.0);
      }
    } catch (e) {
      _showErrorSnackBar("Failed to refresh location");
    }
  }

  // === NEARBY STREETS (OPTIMIZED WITH CACHING) ===

  final Map<String, List<String>> _streetCache = {}; // Cache by lat,lon key

  /// Fetch nearby streets with caching
  Future<List<String>> _fetchNearbyStreets(
    double lat,
    double lon,
    int radius,
  ) async {
    // Create cache key (rounded to 3 decimals ≈ 100m precision)
    final cacheKey = '${lat.toStringAsFixed(3)},${lon.toStringAsFixed(3)}';

    // Return cached result if available
    if (_streetCache.containsKey(cacheKey)) {
      return _streetCache[cacheKey]!;
    }

    final overpassQuery = """
      [out:json][timeout:10];
      (
        way(around:$radius,$lat,$lon)["highway"];
      );
      out tags;
    """;

    try {
      final response = await http
          .post(
            Uri.parse('https://overpass-api.de/api/interpreter'),
            body: overpassQuery,
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        List<String> streets = [];
        for (var element in data['elements']) {
          if (element['tags'] != null && element['tags']['name'] != null) {
            streets.add(element['tags']['name']);
          }
        }
        final uniqueStreets = streets.toSet().toList()..sort();

        // Cache result
        _streetCache[cacheKey] = uniqueStreets;

        return uniqueStreets;
      }
    } catch (e) {
      print("Overpass API error: $e");
    }

    return [];
  }

  // === PIN LOCATION ===

  /// Convert address to LatLng for pin (with debouncing)
  Future<void> _pinLocationFromAddress(String address) async {
    if (address.trim().isEmpty) return;

    try {
      List<Location> locations = await locationFromAddress(
        address,
      ).timeout(const Duration(seconds: 5));

      if (locations.isNotEmpty) {
        final latLng = LatLng(
          locations.first.latitude,
          locations.first.longitude,
        );
        setState(() {
          _pinnedLatLng = latLng;
        });
        _mapController.move(latLng, 15.0);
      } else {
        _showErrorSnackBar("Location not found");
      }
    } catch (e) {
      print("Error converting address: $e");
      _showErrorSnackBar("Failed to find location");
    }
  }

  // === UI HELPERS ===

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 3),
        ),
      );
    }
  }

  Timer? _debounce;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Map
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: _currentLatLng ?? LatLng(14.5995, 120.9842),
              initialZoom: 15.0,
              maxZoom: 18.0,
              minZoom: 10.0,
            ),
            children: [
              TileLayer(
                urlTemplate:
                    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                subdomains: const ['a', 'b', 'c'],
                userAgentPackageName: 'com.example.ark_nav',
                tileProvider: NetworkTileProvider(), // Use default caching
              ),

              // Current Location
              if (_currentLatLng != null) ...[
                CircleLayer(
                  circles: [
                    CircleMarker(
                      point: _currentLatLng!,
                      radius: 300,
                      useRadiusInMeter: true,
                      color: Colors.blue.withOpacity(0.1),
                      borderStrokeWidth: 2,
                      borderColor: Colors.blue,
                    ),
                  ],
                ),
                MarkerLayer(
                  markers: [
                    Marker(
                      point: _currentLatLng!,
                      width: 50,
                      height: 50,
                      child: GestureDetector(
                        onTap:
                            () => showFloodForecastDialog(
                              context,
                              _currentLatLng!,
                              locationLabel: _locationController.text,
                            ),
                        child: const Icon(
                          Icons.location_on,
                          color: Color(0xFF01AFBA),
                          size: 40,
                        ),
                      ),
                    ),
                  ],
                ),
              ],

              // Pinned Location
              if (_pinnedLatLng != null) ...[
                CircleLayer(
                  circles: [
                    CircleMarker(
                      point: _pinnedLatLng!,
                      radius: 300,
                      useRadiusInMeter: true,
                      color: Colors.red.withOpacity(0.1),
                      borderStrokeWidth: 2,
                      borderColor: Colors.red,
                    ),
                  ],
                ),
                MarkerLayer(
                  markers: [
                    Marker(
                      point: _pinnedLatLng!,
                      width: 50,
                      height: 50,
                      child: GestureDetector(
                        onTap:
                            () => showFloodForecastDialog(
                              context,
                              _pinnedLatLng!,
                              locationLabel: _locationController.text,
                            ),
                        child: const Icon(
                          Icons.location_on,
                          color: Colors.red,
                          size: 40,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ],
          ),

          // Top fade overlay
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            height: 120,
            child: Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.black54, Colors.transparent],
                ),
              ),
            ),
          ),

          // Top bar
          Positioned(
            top: 40,
            left: 16,
            right: 16,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  width: 50,
                  height: 50,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    color: Color(0xFF1D364E),
                    image: DecorationImage(
                      image: AssetImage('assets/logo.png'),
                      fit: BoxFit.cover,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 4,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1D364E),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: const [
                        BoxShadow(
                          color: Colors.black26,
                          blurRadius: 8,
                          offset: Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _statusCircle(Colors.green, 'Safe'),
                        _statusCircle(Colors.yellow, 'Moderate'),
                        _statusCircle(Colors.red, 'Danger'),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Column(
                  children: [
                    _topIconButton(Icons.settings, () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const Settings(),
                        ),
                      );
                    }),
                    const SizedBox(height: 8),
                    _topIconButton(Icons.notifications, () {}),
                    const SizedBox(width: 8),
                  ],
                ),
              ],
            ),
          ),

          // Bottom UI
          Align(
            alignment: Alignment.bottomCenter,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      _circleIconButton(
                        Icons.my_location,
                        _refreshCurrentLocation,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: const BoxDecoration(
                    color: Color(0xFF1D364E),
                    image: DecorationImage(
                      image: AssetImage('assets/wave.png'),
                      fit: BoxFit.cover,
                      opacity: 0.5,
                    ),
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(30),
                      topRight: Radius.circular(30),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 8,
                        offset: Offset(0, -2),
                      ),
                    ],
                  ),
                  child: SafeArea(
                    top: false,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // Duration selector
                        Container(
                          padding: const EdgeInsets.symmetric(
                            vertical: 8,
                            horizontal: 4,
                          ),
                          margin: const EdgeInsets.only(bottom: 16),
                          decoration: BoxDecoration(
                            color: Colors.white12,
                            borderRadius: BorderRadius.circular(30),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            children:
                                ['Now', '3 Days', '7 Days'].map((duration) {
                                  final bool isSelected =
                                      selectedDuration == duration;
                                  return GestureDetector(
                                    onTap: () {
                                      setState(() {
                                        selectedDuration = duration;
                                      });
                                    },
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(
                                        vertical: 8,
                                        horizontal: 16,
                                      ),
                                      decoration: BoxDecoration(
                                        color:
                                            isSelected
                                                ? const Color(0xFF01AFBA)
                                                : Colors.transparent,
                                        borderRadius: BorderRadius.circular(20),
                                      ),
                                      child: Text(
                                        duration,
                                        style: TextStyle(
                                          color: Colors.white,
                                          fontWeight:
                                              isSelected
                                                  ? FontWeight.bold
                                                  : FontWeight.normal,
                                        ),
                                      ),
                                    ),
                                  );
                                }).toList(),
                          ),
                        ),

                        // Current Location input
                        TextField(
                          controller: _locationController,
                          readOnly: true, // Current location is auto-updated
                          style: const TextStyle(color: Colors.white),
                          decoration: InputDecoration(
                            labelText: 'Current Location',
                            labelStyle: const TextStyle(color: Colors.white),
                            prefixIcon:
                                _isLoadingLocation
                                    ? const SizedBox(
                                      width: 20,
                                      height: 20,
                                      child: Center(
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Color(0xFF01AFBA),
                                        ),
                                      ),
                                    )
                                    : const Icon(
                                      Icons.my_location,
                                      color: Colors.white,
                                    ),
                            suffixIcon: IconButton(
                              icon: const Icon(
                                Icons.remove_red_eye,
                                color: Colors.white,
                              ),
                              hoverColor: Color(0xFF01AFBA),
                              onPressed: () {
                                showFloodForecastDialog(
                                  context,
                                  _currentLatLng!,
                                  locationLabel: _locationController.text,
                                );
                              },
                            ),
                            filled: true,
                            fillColor: const Color(0xFF1D364E),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(20),
                              borderSide: const BorderSide(
                                color: Colors.white70,
                              ),
                            ),
                            enabledBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(20),
                              borderSide: const BorderSide(
                                color: Colors.white70,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 12),

                        // Pin another location input
                        TextField(
                          controller: _pinLocationController,
                          style: const TextStyle(color: Colors.white),
                          onChanged: (value) async {
                            // Cancel the previous timer if it’s still running
                            if (_debounce?.isActive ?? false)
                              _debounce!.cancel();

                            // Start a new timer
                            _debounce = Timer(
                              const Duration(milliseconds: 500),
                              () async {
                                if (value.isNotEmpty) {
                                  final results =
                                      await fetchLocationSuggestions(value);
                                  setState(() => suggestions = results);
                                } else {
                                  setState(() => suggestions = []);
                                }
                              },
                            );
                          },
                          decoration: InputDecoration(
                            labelText: 'Pin another location',
                            labelStyle: const TextStyle(color: Colors.white),
                            hintText: 'Enter address or place name',
                            hintStyle: const TextStyle(color: Colors.white70),
                            prefixIcon: const Icon(
                              Icons.location_on,
                              color: Colors.white,
                            ),
                            suffixIcon: IconButton(
                              icon: const Icon(
                                Icons.remove_red_eye,
                                color: Colors.white,
                              ),
                              hoverColor: Color(0xFF01AFBA),
                              onPressed: () {
                                showFloodForecastDialog(
                                  context,
                                  _pinnedLatLng!,
                                  locationLabel: _locationController.text,
                                );
                              },
                            ),
                            filled: true,
                            fillColor: const Color(0xFF1D364E),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(20),
                              borderSide: const BorderSide(color: Colors.white),
                            ),
                          ),
                        ),

                        if (suggestions.isNotEmpty)
                          Container(
                            decoration: BoxDecoration(
                              color: const Color.fromARGB(127, 83, 83, 83),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: ConstrainedBox(
                              constraints: const BoxConstraints(
                                maxHeight:
                                    150, // 🔹 Set your desired max height (in pixels)
                              ),
                              child: ListView.builder(
                                padding:
                                    EdgeInsets
                                        .zero, // Optional: remove extra padding
                                itemCount: suggestions.length,
                                itemBuilder: (context, index) {
                                  final place = suggestions[index];
                                  return ListTile(
                                    leading: const Icon(
                                      Icons.location_on,
                                      color: Color(0xFF01AFBA),
                                    ),
                                    title: Text(
                                      place['display_name'],
                                      style: const TextStyle(
                                        color: Colors.white,
                                      ),
                                    ),
                                    onTap: () {
                                      _pinLocationController.text =
                                          place['display_name'];
                                      setState(() => suggestions = []);
                                      _pinLocationFromAddress(
                                        place['display_name'],
                                      );
                                    },
                                  );
                                },
                              ),
                            ),
                          ),

                        const SizedBox(height: 16),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _topIconButton(IconData icon, VoidCallback onPressed) {
    return Container(
      width: 45,
      height: 45,
      decoration: const BoxDecoration(
        shape: BoxShape.circle,
        color: Color(0xFF1D364E),
        boxShadow: [
          BoxShadow(color: Colors.black26, blurRadius: 4, offset: Offset(0, 2)),
        ],
      ),
      child: IconButton(
        icon: Icon(icon, color: const Color(0xFF01AFBA)),
        onPressed: onPressed,
      ),
    );
  }

  Widget _circleIconButton(IconData icon, VoidCallback onPressed) {
    return Container(
      width: 50,
      height: 50,
      decoration: const BoxDecoration(
        shape: BoxShape.circle,
        color: Color(0xFF1D364E),
        boxShadow: [
          BoxShadow(color: Colors.black26, blurRadius: 4, offset: Offset(0, 2)),
        ],
      ),
      child: IconButton(
        icon: Icon(icon, color: const Color(0xFF01AFBA)),
        onPressed: onPressed,
      ),
    );
  }

  Widget _statusCircle(Color color, String label) {
    return Column(
      children: [
        CircleAvatar(radius: 12, backgroundColor: color),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(color: Colors.white)),
      ],
    );
  }
}
