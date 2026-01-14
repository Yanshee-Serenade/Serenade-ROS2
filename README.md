# Yanshee Model Server

A modularized server for Yanshee robot that integrates vision-language models, depth estimation, and ROS tracking.

## Project Structure

```
yanshee_model/
├── modules/                    # Modularized components
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration constants
│   ├── models.py             # AI model management
│   ├── ros_client.py         # ROS client and tracking
│   ├── depth_processing.py   # Depth map generation and analysis
│   ├── text_generation.py    # VLM text streaming
│   ├── app.py               # Flask application
│   └── main.py              # Main entry point
├── server.py                # Main server file (entry point)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── YanAPI.py               # Yanshee RESTful API wrapper (external)
├── ros_api.py              # ROS API interface
├── images/                 # Generated images directory
└── walker/                 # Walker-related modules
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Vision-Language Model Integration**: Supports Qwen, SmolVLM, and other VLMs
- **Depth Estimation**: Depth Anything 3 (DA3) for depth map generation
- **ROS Integration**: Real-time tracking and point cloud data from ROS
- **Streaming Text Generation**: Server-Sent Events (SSE) for real-time text streaming
- **Depth Comparison**: Compare ORB-SLAM3 depths with DA3 predictions
- **RESTful API**: Flask-based API with CORS support

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yanshee_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure ROS API is available:
   - The `ros_api.py` module should be in the project root
   - ROS server should be running at `127.0.0.1:51121`

## Usage

### Starting the Server

```bash
python server.py
```

The server will:
1. Initialize AI models (VLM and DA3)
2. Start Flask application on `http://0.0.0.0:51122`
3. Provide health check endpoint at `/health`

### API Endpoints

- `GET /health` - Health check and model status
- `POST /generate` - Generate text from image with depth processing
- `GET /test` - Test endpoint for debugging

### Generate Endpoint

**Request:**
```json
POST /generate
Content-Type: application/json

{
    "text": "Describe this image"
}
```

**Response:**
Server-Sent Events (SSE) stream with generated text.

## Configuration

Edit `modules/config.py` to modify:

- **Model paths**: Change default VLM or DA3 models
- **Network settings**: Adjust ROS and Flask server addresses
- **Generation parameters**: Modify token limits and save paths
- **Image processing**: Configure depth map resolution and methods

## Modules Overview

### `config.py`
Contains all configuration constants and type aliases.

### `models.py`
Manages AI model lifecycle:
- VLM (Vision Language Model) loading
- DA3 (Depth Anything 3) initialization
- SAM3 (Segment Anything 3) support
- Model cleanup and resource management

### `ros_client.py`
Handles ROS communication:
- Tracking client initialization
- Image and point cloud data retrieval
- Point cloud validation and processing

### `depth_processing.py`
Depth map operations:
- DA3 depth map generation
- Depth visualization and saving
- Depth comparison with ORB-SLAM3
- Statistical analysis

### `text_generation.py`
Text generation with VLMs:
- Streaming text generation
- Batch text generation
- Input validation and formatting
- Error handling

### `app.py`
Flask application:
- Route registration
- Request handling
- Error response formatting
- Server management

### `main.py`
Main entry point:
- Model initialization
- Application startup
- Cleanup handling
- Error management

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- **Flask**: Web framework
- **Transformers**: Hugging Face model loading
- **Depth Anything 3**: Depth estimation
- **OpenCV**: Image processing
- **PyTorch**: Deep learning framework
- **Matplotlib**: Plotting and visualization

## Error Handling

The server includes comprehensive error handling:

1. **Model initialization errors**: Graceful degradation
2. **ROS connection errors**: Informative error messages
3. **Request validation**: Input validation with clear feedback
4. **Streaming errors**: Proper SSE error formatting

## Development

### Adding New Models

1. Add model loading logic to `models.py`
2. Update configuration in `config.py`
3. Add corresponding processing logic in relevant modules

### Extending Functionality

1. Create new module in `modules/` directory
2. Import and integrate with existing modules
3. Update `app.py` to expose new endpoints if needed

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Check code quality
flake8 modules/
mypy modules/
```

## Troubleshooting

### Common Issues

1. **Model loading failures**: Check model paths and dependencies
2. **ROS connection issues**: Verify ROS server is running
3. **Memory errors**: Reduce model size or enable GPU memory optimization
4. **Import errors**: Ensure all dependencies are installed

### Logs

Check console output for detailed logs including:
- Model loading status
- ROS connection attempts
- Request processing steps
- Error messages with timestamps

## License

[Add appropriate license information]

## Acknowledgments

- Yanshee robot platform
- Hugging Face for model hosting
- Depth Anything team for DA3
- ROS community for robotics middleware