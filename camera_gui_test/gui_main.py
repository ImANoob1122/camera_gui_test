import rclpy
from rclpy.node import Node
import tkinter as tk
from PIL import Image, ImageTk
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2

# Import actual message types from image_detector
try:
    from image_detector.msg import LineSegmentArray, BallPositionArray, LineSegment, BallPosition
    DETECTOR_MSGS_AVAILABLE = True
except ImportError:
    DETECTOR_MSGS_AVAILABLE = False
    print("Warning: image_detector.msg not found. Detection features will be disabled.")
    print("Make sure to build and source the image_detector package.")

class CameraGuiApp(Node):
    def __init__(self, root):
        super().__init__('camera_gui_node')
        self.root = root
        self.root.title("Camera & Detection Viewer")
        self.bridge = CvBridge()
        self.is_running = True

        self.current_cv_image = None
        self.detected_lines = []
        self.detected_balls = []

        # GUI Layout
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.root, width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_frame.pack_propagate(False)

        # Image Display
        self.image_label = tk.Label(self.image_frame, text="Waiting for Image...", bg="gray20", fg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Motion>", self.show_pixel_info)

        # Control Panel Title
        tk.Label(self.control_frame, text="Control Panel", font=("Arial", 14, "bold")).pack(pady=10)

        # Image Topic Selection
        tk.Label(self.control_frame, text="Image Topic:", font=("Arial", 10)).pack(pady=(10,0), padx=10, anchor=tk.W)
        self.image_topic_var = tk.StringVar(self.root)
        self.image_topic_var.set("/camera/resize_hsv/image_raw")
        self.topic_options = ["/camera/resize_hsv/image_raw", "/camera/image_raw"]
        self.topic_menu = tk.OptionMenu(self.control_frame, self.image_topic_var, *self.topic_options)
        self.topic_menu.pack(pady=5, padx=10, fill=tk.X)
        self.image_topic_var.trace_add("write", self.on_topic_change)

        # Detection Controls
        tk.Label(self.control_frame, text="Detection Overlays:", font=("Arial", 10)).pack(pady=(20,5), padx=10, anchor=tk.W)
        
        # Line Detection Checkbox
        self.show_lines_var = tk.BooleanVar(value=False)
        self.show_lines_check = tk.Checkbutton(
            self.control_frame, 
            text="Show Detected Lines", 
            variable=self.show_lines_var,
            state=tk.NORMAL if DETECTOR_MSGS_AVAILABLE else tk.DISABLED
        )
        self.show_lines_check.pack(pady=5, padx=20, anchor=tk.W)
        if DETECTOR_MSGS_AVAILABLE:
            self.show_lines_var.trace_add("write", self.on_show_lines_change)

        # Ball Detection Checkbox
        self.show_balls_var = tk.BooleanVar(value=False)
        self.show_balls_check = tk.Checkbutton(
            self.control_frame, 
            text="Show Detected Balls", 
            variable=self.show_balls_var,
            state=tk.NORMAL if DETECTOR_MSGS_AVAILABLE else tk.DISABLED
        )
        self.show_balls_check.pack(pady=5, padx=20, anchor=tk.W)
        if DETECTOR_MSGS_AVAILABLE:
            self.show_balls_var.trace_add("write", self.on_show_balls_change)

        # Detection Status
        tk.Label(self.control_frame, text="Detection Status:", font=("Arial", 10)).pack(pady=(20,5), padx=10, anchor=tk.W)
        self.detection_status_text = tk.Text(self.control_frame, height=6, width=30, wrap=tk.WORD)
        self.detection_status_text.pack(pady=5, padx=10, fill=tk.X)
        self.detection_status_text.config(state=tk.DISABLED)

        # Pixel Info Display
        tk.Label(self.control_frame, text="Pixel Information:", font=("Arial", 10)).pack(pady=(20,5), padx=10, anchor=tk.W)
        self.pixel_info_frame = tk.Frame(self.control_frame, relief=tk.SUNKEN, borderwidth=1)
        self.pixel_info_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.pixel_coord_label = tk.Label(self.pixel_info_frame, text="Position: (N/A)", anchor=tk.W)
        self.pixel_coord_label.pack(padx=5, pady=2, fill=tk.X)
        
        self.pixel_bgr_label = tk.Label(self.pixel_info_frame, text="BGR: (N/A)", anchor=tk.W)
        self.pixel_bgr_label.pack(padx=5, pady=2, fill=tk.X)
        
        self.pixel_hsv_label = tk.Label(self.pixel_info_frame, text="HSV: (N/A)", anchor=tk.W)
        self.pixel_hsv_label.pack(padx=5, pady=2, fill=tk.X)

        # Initialize subscribers
        self.image_subscriber = None
        self.line_subscriber = None
        self.ball_subscriber = None

        self.create_image_subscriber(self.image_topic_var.get())
        if DETECTOR_MSGS_AVAILABLE:
            self.on_show_lines_change()
            self.on_show_balls_change()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Update detection status
        self.update_detection_status()

    def update_detection_status(self):
        """Update the detection status display"""
        self.detection_status_text.config(state=tk.NORMAL)
        self.detection_status_text.delete(1.0, tk.END)
        
        if not DETECTOR_MSGS_AVAILABLE:
            self.detection_status_text.insert(tk.END, "Detection unavailable!\nBuild image_detector package.")
        else:
            status_lines = []
            status_lines.append(f"Lines: {len(self.detected_lines)} detected")
            status_lines.append(f"Balls: {len(self.detected_balls)} detected")
            
            if self.detected_balls:
                status_lines.append("\nBall colors:")
                for ball in self.detected_balls:
                    status_lines.append(f"  - {ball.color}")
            
            self.detection_status_text.insert(tk.END, "\n".join(status_lines))
        
        self.detection_status_text.config(state=tk.DISABLED)

    def show_pixel_info(self, event):
        """Display pixel information at mouse position"""
        if self.current_cv_image is None:
            self.pixel_coord_label.config(text="Position: (N/A)")
            self.pixel_bgr_label.config(text="BGR: (N/A)")
            self.pixel_hsv_label.config(text="HSV: (N/A)")
            return

        if not hasattr(self.image_label, 'image') or self.image_label.image is None:
            return

        # Get image display dimensions
        img_display_width = self.image_label.image.width()
        img_display_height = self.image_label.image.height()

        # Get label dimensions
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        # Calculate offset (centered image)
        offset_x = (label_width - img_display_width) // 2
        offset_y = (label_height - img_display_height) // 2

        # Convert mouse coordinates to image coordinates
        img_x = event.x - offset_x
        img_y = event.y - offset_y

        # Check bounds
        if 0 <= img_x < img_display_width and 0 <= img_y < img_display_height:
            cv_height, cv_width = self.current_cv_image.shape[:2]
            
            # Scale coordinates if needed
            scale_x = cv_width / img_display_width
            scale_y = cv_height / img_display_height
            cv_x = int(img_x * scale_x)
            cv_y = int(img_y * scale_y)
            
            if 0 <= cv_y < cv_height and 0 <= cv_x < cv_width:
                # Get BGR value
                bgr_color = self.current_cv_image[cv_y, cv_x]
                
                # Convert to HSV
                hsv_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2HSV)
                hsv_color = hsv_image[cv_y, cv_x]
                
                # Update labels
                self.pixel_coord_label.config(text=f"Position: ({cv_x}, {cv_y})")
                self.pixel_bgr_label.config(text=f"BGR: ({bgr_color[0]}, {bgr_color[1]}, {bgr_color[2]})")
                self.pixel_hsv_label.config(text=f"HSV: ({hsv_color[0]}, {hsv_color[1]}, {hsv_color[2]})")
            else:
                self.pixel_coord_label.config(text="Position: (Out of bounds)")
                self.pixel_bgr_label.config(text="BGR: (N/A)")
                self.pixel_hsv_label.config(text="HSV: (N/A)")
        else:
            self.pixel_coord_label.config(text="Position: (Outside image)")
            self.pixel_bgr_label.config(text="BGR: (N/A)")
            self.pixel_hsv_label.config(text="HSV: (N/A)")

    def on_topic_change(self, *args):
        """Handle image topic change"""
        selected_topic = self.image_topic_var.get()
        self.get_logger().info(f"Image topic changed to: {selected_topic}")
        self.create_image_subscriber(selected_topic)
        self.current_cv_image = None
        self.image_label.config(image=None, text="Waiting for Image...")
        self.image_label.image = None
        self.detected_lines = []
        self.detected_balls = []
        self.update_detection_status()

    def create_image_subscriber(self, topic_name):
        """Create or recreate image subscriber"""
        if self.image_subscriber is not None:
            self.destroy_subscription(self.image_subscriber)
        self.image_subscriber = self.create_subscription(
            RosImage, topic_name, self.image_callback, 10)
        self.get_logger().info(f"Image subscriber created for '{topic_name}'")

    def image_callback(self, msg):
        """Handle incoming image messages"""
        try:
            self.current_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def on_show_lines_change(self, *args):
        """Handle line detection checkbox change"""
        if not DETECTOR_MSGS_AVAILABLE:
            return
            
        if self.show_lines_var.get():
            if self.line_subscriber is None:
                self.line_subscriber = self.create_subscription(
                    LineSegmentArray, "/detection/lines", self.line_detection_callback, 10)
                self.get_logger().info("Subscribed to /detection/lines")
        else:
            if self.line_subscriber is not None:
                self.destroy_subscription(self.line_subscriber)
                self.line_subscriber = None
                self.get_logger().info("Unsubscribed from /detection/lines")
            self.detected_lines = []
            self.update_detection_status()

    def line_detection_callback(self, msg):
        """Handle line detection messages"""
        self.detected_lines = msg.lines
        self.get_logger().debug(f'Received {len(msg.lines)} lines')
        self.update_detection_status()

    def on_show_balls_change(self, *args):
        """Handle ball detection checkbox change"""
        if not DETECTOR_MSGS_AVAILABLE:
            return
            
        if self.show_balls_var.get():
            if self.ball_subscriber is None:
                self.ball_subscriber = self.create_subscription(
                    BallPositionArray, "/detection/balls", self.ball_detection_callback, 10)
                self.get_logger().info("Subscribed to /detection/balls")
        else:
            if self.ball_subscriber is not None:
                self.destroy_subscription(self.ball_subscriber)
                self.ball_subscriber = None
                self.get_logger().info("Unsubscribed from /detection/balls")
            self.detected_balls = []
            self.update_detection_status()

    def ball_detection_callback(self, msg):
        """Handle ball detection messages"""
        self.detected_balls = msg.balls
        self.get_logger().debug(f'Received {len(msg.balls)} balls')
        self.update_detection_status()

    def update_gui_display(self):
        """Update the GUI display with current image and detections"""
        if self.current_cv_image is not None:
            display_image = self.current_cv_image.copy()

            # Draw detected lines
            if self.show_lines_var.get() and self.detected_lines:
                for line_segment in self.detected_lines:
                    start_point = (int(line_segment.start.x), int(line_segment.start.y))
                    end_point = (int(line_segment.end.x), int(line_segment.end.y))
                    cv2.line(display_image, start_point, end_point, (0, 255, 0), 3)

            # Draw detected balls
            if self.show_balls_var.get() and self.detected_balls:
                for ball_pos in self.detected_balls:
                    center = (int(ball_pos.position.x), int(ball_pos.position.y))
                    
                    # Color mapping based on ball color name
                    color_bgr = (0, 0, 255)  # Default red
                    if ball_pos.color.lower() == "blue":
                        color_bgr = (255, 0, 0)
                    elif ball_pos.color.lower() == "yellow":
                        color_bgr = (0, 255, 255)
                    elif ball_pos.color.lower() == "green":
                        color_bgr = (0, 255, 0)
                    elif ball_pos.color.lower() == "red":
                        color_bgr = (0, 0, 255)

                    # Draw circle and label
                    cv2.circle(display_image, center, 15, color_bgr, -1)
                    cv2.circle(display_image, center, 15, (0, 0, 0), 2)  # Black outline
                    
                    # Add text label
                    text_pos = (center[0] - 20, center[1] + 30)
                    cv2.putText(display_image, ball_pos.color,
                              text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_image, ball_pos.color,
                              text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Convert to RGB and display
            try:
                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Resize if too large
                max_width = 800
                max_height = 600
                if pil_image.width > max_width or pil_image.height > max_height:
                    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                tk_image = ImageTk.PhotoImage(image=pil_image)
                self.image_label.config(image=tk_image, text="")
                self.image_label.image = tk_image
            except Exception as e:
                self.get_logger().error(f"Error updating display: {e}")

    def update_ros_and_gui(self):
        """Main update loop"""
        if not self.is_running:
            return
        rclpy.spin_once(self, timeout_sec=0.01)
        self.update_gui_display()
        if self.is_running:
            self.root.after(50, self.update_ros_and_gui)

    def on_closing(self):
        """Handle window close event"""
        self.get_logger().info("Shutting down...")
        self.is_running = False

        # Clean up subscribers
        if self.image_subscriber:
            self.destroy_subscription(self.image_subscriber)
        if self.line_subscriber:
            self.destroy_subscription(self.line_subscriber)
        if self.ball_subscriber:
            self.destroy_subscription(self.ball_subscriber)

        # Destroy node and window
        if rclpy.ok() and self.is_valid():
            self.destroy_node()
        self.root.destroy()

def main(args=None):
    rclpy.init(args=args)
    
    root = tk.Tk()
    root.geometry("1100x700")
    app = CameraGuiApp(root)

    try:
        app.update_ros_and_gui()
        app.root.mainloop()
    except Exception as e:
        if app.is_valid():
            app.get_logger().error(f"Unhandled exception: {e}")
        else:
            print(f"Exception after shutdown: {e}")
    finally:
        app.is_running = False
        if app.is_valid():
            app.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()