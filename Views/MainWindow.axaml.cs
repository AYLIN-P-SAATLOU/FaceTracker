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
    
    // THE PADLOCK to prevent crashes
    private readonly object _cameraLock = new object();
    
    private Image? _cameraDisplayControl; 
    private TextBlock? _faceCountLabel;
    private bool _applyBlur = false; 

    // UPGRADED MODEL: Make sure you download this new XML file!
    private CascadeClassifier _faceCascade = new CascadeClassifier("haarcascade_frontalface_alt2.xml");

    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this); 
        
        _cameraDisplayControl = this.FindControl<Image>("CameraDisplay");
        _faceCountLabel = this.FindControl<TextBlock>("FaceCountLabel");

        var startButton = this.FindControl<Button>("StartButton");
        if (startButton != null) startButton.Click += OnStartCameraClick;

        var stopButton = this.FindControl<Button>("StopButton");
        if (stopButton != null) stopButton.Click += OnStopCameraClick;

        var blurCheckBox = this.FindControl<CheckBox>("BlurCheckBox");
        if (blurCheckBox != null)
        {
            blurCheckBox.IsCheckedChanged += (s, e) => _applyBlur = blurCheckBox.IsChecked ?? false;
        }

        _timer = new System.Timers.Timer(33); 
        _timer.Elapsed += (s, e) => GrabFrame();
    }

    private void OnStartCameraClick(object? sender, RoutedEventArgs e)
    {
        lock (_cameraLock) // Lock before creating
        {
            if (_capture != null) return; 

            _capture = new VideoCapture(0); 
            if (!_capture.IsOpened()) return;
        }
        
        _timer.Start();
    }

    private void OnStopCameraClick(object? sender, RoutedEventArgs e)
    {
        _timer.Stop();

        // Lock before destroying so we don't crash the GrabFrame thread
        lock (_cameraLock)
        {
            if (_capture != null)
            {
                _capture.Release();
                _capture.Dispose();
                _capture = null;
            }
        }

        Dispatcher.UIThread.InvokeAsync(() =>
        {
            if (_faceCountLabel != null) 
                _faceCountLabel.Text = "Camera Stopped (Frame Frozen)";
        });
    }

    private void GrabFrame()
    {
        if (_isProcessing) return;
        _isProcessing = true;

        try 
        {
            using var frame = new Mat();
            
            // Lock while reading from the camera
            lock (_cameraLock)
            {
                if (_capture == null || !_capture.IsOpened()) return;
                if (!_capture.Read(frame) || frame.Empty()) return;
            }

            // Processing the frame happens outside the lock to keep it fast
            using var gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

            var faces = _faceCascade.DetectMultiScale(
                gray, 
                scaleFactor: 1.05, 
                minNeighbors: 3, 
                flags: HaarDetectionTypes.ScaleImage, 
                minSize: new OpenCvSharp.Size(40, 40));

            foreach (var rect in faces)
            {
                if (_applyBlur)
                {
                    using var faceRoi = new Mat(frame, rect);
                    int blurStrength = (rect.Width / 4) * 2 + 1; 
                    Cv2.GaussianBlur(faceRoi, faceRoi, new OpenCvSharp.Size(blurStrength, blurStrength), 0);
                    Cv2.Rectangle(frame, rect, Scalar.Gray, 2);
                }
                else
                {
                    Cv2.Rectangle(frame, rect, Scalar.Green, 2);
                }
            }

            var bitmap = ConvertMatToBitmap(frame);
            int count = faces.Length;

            Dispatcher.UIThread.InvokeAsync(() =>
            {
                if (_cameraDisplayControl != null) _cameraDisplayControl.Source = bitmap;
                if (_faceCountLabel != null) _faceCountLabel.Text = $"Faces Detected: {count}";
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
        _faceCascade?.Dispose();
        base.OnClosed(e);
    }
}