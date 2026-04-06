using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Markup.Xaml;
using OpenCvSharp;
using System;
using Avalonia.Threading;
using System.Collections.Generic;

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

    // The two main models for front and side face detection
    private CascadeClassifier _frontalFace = new CascadeClassifier("haarcascade_frontalface_alt2.xml");
    private CascadeClassifier _profileFace = new CascadeClassifier("haarcascade_profileface.xml");

    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this); 
        _cameraDisplayControl = this.FindControl<Image>("CameraDisplay");
        _faceCountLabel = this.FindControl<TextBlock>("FaceCountLabel");

        this.FindControl<Button>("StartButton")!.Click += OnStartCameraClick;
        this.FindControl<Button>("StopButton")!.Click += OnStopCameraClick;

        var blurCheckBox = this.FindControl<CheckBox>("BlurCheckBox");
        if (blurCheckBox != null)
            blurCheckBox.IsCheckedChanged += (s, e) => _applyBlur = blurCheckBox.IsChecked ?? false;

        this.Opened += (s, e) => { this.Activate(); this.Topmost = true; };

        _timer = new System.Timers.Timer(33); 
        _timer.Elapsed += (s, e) => GrabFrame();
    }

    private void OnStartCameraClick(object? sender, RoutedEventArgs e)
    {
        lock (_cameraLock)
        {
            if (_capture != null) return; 
            
            _capture = new VideoCapture(0); 
            
            // SPEED FIX: Force the camera to 640x480 resolution. 
            // This stops 1080p/4K lag and makes the AI run instantly.
            _capture.Set(VideoCaptureProperties.FrameWidth, 640);
            _capture.Set(VideoCaptureProperties.FrameHeight, 480);
            
            if (!_capture.IsOpened()) return;
        }
        _timer.Start();
    }

    private void OnStopCameraClick(object? sender, RoutedEventArgs e)
    {
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
            Cv2.EqualizeHist(gray, gray); 

            // SPEED FIX: ScaleFactor set to 1.2 for faster scanning
            OpenCvSharp.Rect[] frontal = _frontalFace.DetectMultiScale(gray, 1.2, 5, HaarDetectionTypes.ScaleImage, new OpenCvSharp.Size(60, 60));
            OpenCvSharp.Rect[] profile = _profileFace.DetectMultiScale(gray, 1.2, 5, HaarDetectionTypes.ScaleImage, new OpenCvSharp.Size(60, 60));

            var allFaces = new List<OpenCvSharp.Rect>(frontal);

            // Deduplication logic
            foreach (var pRect in profile)
            {
                bool isDuplicate = false;
                foreach (var fRect in frontal)
                {
                    OpenCvSharp.Rect intersect = pRect & fRect; 
                    if (intersect.Width > 0 && intersect.Height > 0)
                    {
                        isDuplicate = true;
                        break; 
                    }
                }
                if (!isDuplicate) allFaces.Add(pRect);
            }

            // Draw bounding boxes or apply blur
            foreach (var faceRect in allFaces)
            {
                if (_applyBlur)
                {
                    using var faceRoi = new Mat(frame, faceRect);
                    
                    // Dynamic blur size based on face width
                    int blurSize = faceRect.Width / 2;
                    if (blurSize % 2 == 0) blurSize++; 

                    // Extreme blur applied
                    Cv2.GaussianBlur(faceRoi, faceRoi, new OpenCvSharp.Size(blurSize, blurSize), 15);
                    
                    Cv2.Rectangle(frame, faceRect, Scalar.Gray, 2);
                }
                else
                {
                    Cv2.Rectangle(frame, faceRect, Scalar.Green, 2);
                }
            }

            var bitmap = ConvertMatToBitmap(frame);
            var count = allFaces.Count;

            Dispatcher.UIThread.InvokeAsync(() =>
            {
                if (_cameraDisplayControl != null) _cameraDisplayControl.Source = bitmap;
                if (_faceCountLabel != null) _faceCountLabel.Text = $"Faces Detected: {count}";
            });
        }
        catch (Exception ex) 
        { 
            Console.WriteLine(ex.Message); 
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
        
        _frontalFace.Dispose();
        _profileFace.Dispose();
        
        base.OnClosed(e);
    }
}