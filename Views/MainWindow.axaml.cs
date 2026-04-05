using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Markup.Xaml;
using OpenCvSharp;
using System;
using Avalonia.Threading;

namespace FaceRecognizerApp.Views;

public partial class MainWindow : Avalonia.Controls.Window
{
    // --- Fields ---
    private VideoCapture? _capture;
    private readonly System.Timers.Timer _timer;
    private bool _isProcessing = false; 
    private Image? _cameraDisplayControl; 

    // The Face Detector Model
    private CascadeClassifier _faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this); 
        
        _cameraDisplayControl = this.FindControl<Image>("CameraDisplay");

        var startButton = this.FindControl<Button>("StartButton");
        if (startButton != null)
        {
            startButton.Click += OnStartCameraClick;
        }

        _timer = new System.Timers.Timer(33); 
        _timer.Elapsed += (s, e) => GrabFrame();
    }

    private void OnStartCameraClick(object? sender, RoutedEventArgs e)
    {
        if (_capture != null) return; 

        _capture = new VideoCapture(0); 
        if (!_capture.IsOpened())
        {
            Console.WriteLine("Could not open camera!");
            return;
        }
        
        _timer.Start();
    }

    private void GrabFrame()
    {
        if (_isProcessing || _capture == null) return;
        _isProcessing = true;

        try 
        {
            using var frame = new Mat();
            if (_capture.Read(frame) && !frame.Empty())
            {
                // --- FACE DETECTION ---
                using var gray = new Mat();
                Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

                var faces = _faceCascade.DetectMultiScale(
                    gray, 
                    scaleFactor: 1.1, 
                    minNeighbors: 5, 
                    minSize: new OpenCvSharp.Size(30, 30));

                // Draw the green box
                foreach (var rect in faces)
                {
                    Cv2.Rectangle(frame, rect, Scalar.Green, 2);
                }

                // --- UPDATE UI ---
                var bitmap = ConvertMatToBitmap(frame);
                Dispatcher.UIThread.InvokeAsync(() =>
                {
                    if (_cameraDisplayControl != null)
                        _cameraDisplayControl.Source = bitmap;
                });
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error processing frame: {ex.Message}");
        }
        finally
        {
            _isProcessing = false;
        }
    }

    private WriteableBitmap ConvertMatToBitmap(Mat mat)
    {
        int width = mat.Width;
        int height = mat.Height;

        using var rgbaMat = new Mat();
        Cv2.CvtColor(mat, rgbaMat, ColorConversionCodes.BGR2BGRA);

        var bitmap = new WriteableBitmap(
            new PixelSize(width, height), 
            new Vector(96, 96), 
            Avalonia.Platform.PixelFormat.Bgra8888, 
            Avalonia.Platform.AlphaFormat.Premul);

        using (var fb = bitmap.Lock())
        {
            unsafe
            {
                Buffer.MemoryCopy(
                    (void*)rgbaMat.Data, 
                    (void*)fb.Address, 
                    (long)width * height * 4, 
                    (long)width * height * 4);
            }
        }
        return bitmap;
    }

    protected override void OnClosed(EventArgs e)
    {
        _timer.Stop();
        _timer.Dispose();
        _capture?.Dispose();
        _faceCascade?.Dispose();
        base.OnClosed(e);
    }
}