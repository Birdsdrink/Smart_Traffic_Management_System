from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.mail import EmailMessage
from django.utils import timezone
from django.urls import reverse
#from .camera import VideoCamera
from .models import *
#from .speed_estimator import *
#from .roboflow_inference import *
from ultralytics import YOLO
import cv2
import torch


camera = None


#cap = cv2.VideoCapture('tc.mp4')
#region_points = [(0, 119), (1018, 119)]

#speed_obj = SpeedEstimator(region=region_points, model="yolo12s.pt", line_width=2)

#cv2.namedWindow("Speed Estimation")
#cv2.setMouseCallback("Speed Estimation", mouse_callback)

# Generating frames, Detecting vehicles, speed and accidents

# Choose device: use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv11n model on the selected device
model = YOLO("yolo11n.pt").to(device)
vehicle_classes = ['car']

# OpenCV VideoCapture (webcam or video)
rtsp_url = "rtsp://admin:12345@41.174.163.212:8554/live/0"
cap = cv2.VideoCapture(rtsp_url)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
                continue

        frame = cv2.resize(frame, (640, 340))
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                if class_name in vehicle_classes:
                    print(f"Detected {class_name} with confidence {box.conf[0]:.2f}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


            



def Home(request):
    return render(request, 'home.html',)


def RegisterView(request):
    if request.method == 'POST':
        # getting user inputs from frontend
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        # Input Valudation
        user_data_has_error = False
        # make sure email and username are not being used

        if User.objects.filter(username=username).exists():
            user_data_has_error = True
            messages.error(request, 'Username already exists')

        if User.objects.filter(email=email).exists():
            user_data_has_error = True
            messages.error(request, 'Email already exists')

        # make aure password is at least 5 characters long
        if len(password) < 5:
            user_data_has_error = True
            messages.error(request, 'Password must be at least 5 characters')
        
        if password!=confirm_password:
            user_data_has_error = True
            messages.error(request, 'Passwords do not match')

        # Create user if there are no errors
        if not user_data_has_error:
            new_user = User.objects.create_user(
                first_name = first_name,
                last_name = last_name,
                email = email,
                username = username,
                password = password
            )
            messages.success(request, 'Account created. Login now')
            return redirect('login')
        else:
            return redirect('register')
    else:
        return render(request, 'register.html')


def LoginView(request):
    if request.method == 'POST':

        # getting user inputs from frontend
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Input Valudation
        user_data_has_error = False

        # make aure password is at least 5 characters long
        if len(password) < 5:
            user_data_has_error = True
            messages.error(request, 'Password must be at least 5 characters')

        # Create user if there are no errors
        user = authenticate(request=request, username=username, password=password)
        if user is not None:
            # login user if login credentials are correct
            login(request, user)

            # ewdirect to home page
            return redirect('dashboard')
        else:
            # redirect back to the login page if credentials are wrong
            messages.error(request, 'Invalid username or password')
            return redirect('login')

    return render(request, 'login.html',)

def LogoutView(request):
    global camera
    if camera:
        camera.release()
        camera = None
    logout(request)
    # redirect to login page after logout
    return redirect('login')

def ResetPasswordView(request, reset_id):
    try:
        password_reset_id = PasswordReset.objects.get(reset_id=reset_id)

        if request.method == "POST":
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')

            passwords_have_error = False

            if password != confirm_password:
                passwords_have_error = True
                messages.error(request, 'Passwords do not match')

            if len(password) < 5:
                passwords_have_error = True
                messages.error(request, 'Password must be at least 5 characters long')

            expiration_time = password_reset_id.created_when + timezone.timedelta(minutes=10)

            if timezone.now() > expiration_time:
                passwords_have_error = True
                messages.error(request, 'Reset link has expired')

                password_reset_id.delete()

            if not passwords_have_error:
                user = password_reset_id.user
                user.set_password(password)
                user.save()

                password_reset_id.delete()

                messages.success(request, 'Password reset. Proceed to login')
                return redirect('login')
            else:
                # redirect back to password reset page and display errors
                return redirect('reset-password', reset_id=reset_id)

    
    except PasswordReset.DoesNotExist:
        
        # redirect to forgot password page if code does not exist
        messages.error(request, 'Invalid reset id')
        return redirect('forgot-password')

    return render(request, 'reset_password.html')

def ForgotPasswordView(request):

    if request.method == "POST":
        email = request.POST.get('email')

        try:
            user = User.objects.get(email=email)

            new_password_reset = PasswordReset(user=user)
            new_password_reset.save()

            password_reset_url = reverse('reset-password', kwargs={'reset_id': new_password_reset.reset_id})

            full_password_reset_url = f'{request.scheme}://{request.get_host()}{password_reset_url}'

            email_body = f'Reset your password using the link below:\n\n\n{full_password_reset_url}'
        
            email_message = EmailMessage(
                'Reset your password', # email subject
                email_body,
                settings.EMAIL_HOST_USER, # email sender
                [email] # email  receiver 
            )

            email_message.fail_silently = True
            email_message.send()

            return redirect('password-reset-sent', reset_id=new_password_reset.reset_id)

        except User.DoesNotExist:
            messages.error(request, f"No user with email '{email}' found")
            return redirect('forgot-password')

    return render(request, 'forgot_password.html')

def PasswordResetSentView(request, reset_id):

    if PasswordReset.objects.filter(reset_id=reset_id).exists():
        return render(request, 'password_reset_sent.html')
    else:
        # redirect to forgot password page if code does not exist
        messages.error(request, 'Invalid reset id')
        return redirect('forgot-password')

@login_required
def DashboardView(request):
    #global camera
    #if not camera:
        #camera = VideoCamera()
    return render(request, 'dashboard.html',)

def VideoFeedView(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
