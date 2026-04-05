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
    private VideoCapture? _capture;
    private readonly System.Timers.Timer _timer;
    private bool _isProcessing = false; 
    private readonly object _cameraLock = new object();
    
    private Image? _cameraDisplayControl; 
    private TextBlock? _faceCountLabel;
    private bool _applyBlur = false; 

    // The two brains for our app: Face and Eyes
    private CascadeClassifier _faceCascade = new CascadeClassifier("haarcascade_frontalface_alt2.xml");
    private CascadeClassifier _eyeCascade = new CascadeClassifier("haarcascade_eye.xml");

    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this); 
        
        _cameraDisplayControl = this.FindControl<Image>("CameraDisplay");
        _faceCountLabel = this.FindControl<TextBlock>("FaceCountLabel");

        this.FindControl<Button>("StartButton")!.Click += OnStartCameraClick;
        this.FindControl<Button>("StopButton")!.Click += OnStopCameraClick;

        var blurCheckBox = this.FindControl<CheckBox>("BlurCheckBox");
        if (blurCheckBox != null)
        {
            blurCheckBox.IsCheckedChanged += (s, e) => _applyBlur = blurCheckBox.IsChecked ?? false;
        }

        // Force window to the front on launch
        this.Opened += (s, e) => 
        {
            this.Activate(); 
            this.Topmost = true; 
        };

        _timer = new System.Timers.Timer(33); // Targeting ~30 FPS
        _timer.Elapsed += (s, e) => GrabFrame();
    }

    private void OnStartCameraClick(object? sender, RoutedEventArgs e)
    {
        lock (_cameraLock)
        {
            if (_capture != null) return; 
            _capture = new VideoCapture(0); 
            if (!_capture.IsOpened()) return;
        }
        _timer.Start();
    }

    private void OnStopCameraClick(object? sender, RoutedEventArgs e)
    {
        // Stop requesting new frames, freezing the UI on the last frame
        _timer.Stop();
        
        lock (_cameraLock)
        {
            if (_capture != null)
            {
                _capture.Release();
                _capture.Dispose();
                _capture = null;
            }
        }
    }

    private void GrabFrame()
    {
        if (_isProcessing) return;
        _isProcessing = true;

        try 
        {
            using var frame = new Mat();
            lock (_cameraLock)
            {
                if (_capture == null || !_capture.IsOpened() || !_capture.Read(frame) || frame.Empty()) return;
            }

            using var gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

            // Core Face Detection
            var faces = _faceCascade.DetectMultiScale(
                gray, 1.05, 3, HaarDetectionTypes.ScaleImage, new OpenCvSharp.Size(40, 40));

            foreach (var faceRect in faces)
            {
                if (_applyBlur)
                {
                    // Privacy Mask (Skip drawing eyes if the face is blurred)
                    using var faceRoi = new Mat(frame, faceRect);
                    int blurStrength = (faceRect.Width / 4) * 2 + 1; 
                    Cv2.GaussianBlur(faceRoi, faceRoi, new OpenCvSharp.Size(blurStrength, blurStrength), 0);
                    Cv2.Rectangle(frame, faceRect, Scalar.Gray, 2);
                }
                else
                {
                    // Standard Tracking Box for Face
                    Cv2.Rectangle(frame, faceRect, Scalar.Green, 2);

                    // Region of Interest (ROI) mapping to just the face area
                    using var faceRoiGray = new Mat(gray, faceRect);
                    using var faceRoiColor = new Mat(frame, faceRect); // Draws directly onto the main frame

                    // Eye Detection inside the face
                    var eyes = _eyeCascade.DetectMultiScale(
                        faceRoiGray, 1.1, 5, HaarDetectionTypes.ScaleImage, new OpenCvSharp.Size(15, 15));
                        
                    foreach (var eye in eyes)
                    {
                        Cv2.Rectangle(faceRoiColor, eye, Scalar.Blue, 1);
                    }
                }
            }

            // Update UI
            var bitmap = ConvertMatToBitmap(frame);
            Dispatcher.UIThread.InvokeAsync(() =>
            {
                if (_cameraDisplayControl != null) _cameraDisplayControl.Source = bitmap;
                if (_faceCountLabel != null) _faceCountLabel.Text = $"Faces Detected: {faces.Length}";
            });
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
        using var rgbaMat = new Mat();
        Cv2.CvtColor(mat, rgbaMat, ColorConversionCodes.BGR2BGRA);
        var bitmap = new WriteableBitmap(new PixelSize(mat.Width, mat.Height), new Vector(96, 96), Avalonia.Platform.PixelFormat.Bgra8888, Avalonia.Platform.AlphaFormat.Premul);
        using (var fb = bitmap.Lock())
        {
            unsafe { Buffer.MemoryCopy((void*)rgbaMat.Data, (void*)fb.Address, (long)mat.Width * mat.Height * 4, (long)mat.Width * mat.Height * 4); }
        }
        return bitmap;
    }

    protected override void OnClosed(EventArgs e)
    {
        _timer.Stop();
        _timer.Dispose();
        
        lock (_cameraLock) 
        { 
            _capture?.Dispose(); 
        }
        
        // Clean up both models
        _faceCascade?.Dispose();
        _eyeCascade?.Dispose();
        
        base.OnClosed(e);
    }
}