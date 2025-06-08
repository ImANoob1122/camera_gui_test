import rclpy
from rclpy.node import Node
import tkinter as tk
from PIL import Image, ImageTk
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2 # For drawing

# Attempt to import actual message types
try:
    from image_detector.msg import LineSegmentArray, BallPositionArray
except ImportError:
    # Fallback to dummy classes if image_detector.msg is not found (e.g. during isolated testing)
    # This allows the GUI to run, but detection subscriptions will not work correctly.
    print("Warning: image_detector.msg not found. Using dummy message types for LineSegmentArray and BallPositionArray.")
    print("Functionality requiring these messages (line/ball detection) will be affected.")

    class PointMsg: # Mimicking geometry_msgs/Point (avoid name clash with tkinter.Point if any)
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class LineSegment: # Mimicking image_detector/LineSegment
        def __init__(self, start_x=0.0, start_y=0.0, end_x=0.0, end_y=0.0):
            self.start = PointMsg(start_x, start_y)
            self.end = PointMsg(end_x, end_y)

    class LineSegmentArray: # Mimicking image_detector/LineSegmentArray
        def __init__(self):
            self.lines = [] # List of LineSegment

    class BallPosition: # Mimicking image_detector/BallPosition
         def __init__(self, x=0.0, y=0.0, color_str="unknown"):
            self.position = PointMsg(x,y)
            self.color = str(color_str)

    class BallPositionArray: # Mimicking image_detector/BallPositionArray
        def __init__(self):
            self.balls = [] # List of BallPosition

class CameraGuiApp(Node):
    def __init__(self, root):
        super().__init__('camera_gui_node')
        self.root = root
        self.root.title("Camera Test GUI")
        self.bridge = CvBridge()
        self.is_running = True # For controlling the update loop

        self.current_cv_image = None
        self.detected_lines = []
        self.detected_balls = []

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.root, width=250)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, text="Waiting for Image...")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Image Topic Selection
        tk.Label(self.control_frame, text="Select Image Topic:").pack(pady=(10,0), padx=5, anchor=tk.W)
        self.image_topic_var = tk.StringVar(self.root)
        self.image_topic_var.set("/camera/crop/image_raw")
        self.topic_options = ["/camera/crop/image_raw", "/camera/hsv/image_raw"]
        self.topic_menu = tk.OptionMenu(self.control_frame, self.image_topic_var, *self.topic_options) # Command is implicitly handled by trace
        self.topic_menu.pack(pady=2, padx=5, fill=tk.X)
        self.image_topic_var.trace_add("write", self.on_topic_change)

        # Line Detection Checkbox
        self.show_lines_var = tk.BooleanVar(value=False) # Default to False
        self.show_lines_check = tk.Checkbutton(self.control_frame, text="Show Lines", variable=self.show_lines_var)
        self.show_lines_check.pack(pady=5, padx=5, anchor=tk.W)
        self.show_lines_var.trace_add("write", self.on_show_lines_change)

        # Ball Detection Checkbox
        self.show_balls_var = tk.BooleanVar(value=False) # Default to False
        self.show_balls_check = tk.Checkbutton(self.control_frame, text="Show Balls", variable=self.show_balls_var)
        self.show_balls_check.pack(pady=5, padx=5, anchor=tk.W)
        self.show_balls_var.trace_add("write", self.on_show_balls_change)

        self.image_subscriber = None
        self.line_subscriber = None
        self.ball_subscriber = None

        # Initialize subscribers based on initial state of variables
        self.create_image_subscriber(self.image_topic_var.get())
        self.on_show_lines_change() # Call to setup subscriber based on initial checkbox state
        self.on_show_balls_change() # Call to setup subscriber based on initial checkbox state

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_topic_change(self, *args): # name, index, mode (from trace)
        selected_topic = self.image_topic_var.get()
        self.get_logger().info(f"Image topic changed to: {selected_topic}")
        self.create_image_subscriber(selected_topic)
        self.current_cv_image = None # Clear old image
        self.image_label.config(image=None) # Clear displayed image
        self.image_label.image = None
        self.image_label.config(text="Waiting for Image on new topic...")
        self.detected_lines = [] # Clear detections
        self.detected_balls = []

    def create_image_subscriber(self, topic_name):
        if self.image_subscriber is not None:
            self.destroy_subscription(self.image_subscriber)
            self.get_logger().info(f"Destroyed previous image subscriber.")
        self.image_subscriber = self.create_subscription(
            RosImage, topic_name, self.image_callback, 10)
        self.get_logger().info(f"Image subscriber created for '{topic_name}'")

    def image_callback(self, msg):
        try:
            self.current_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def on_show_lines_change(self, *args):
        state = self.show_lines_var.get()
        if state:
            if self.line_subscriber is None:
                self.get_logger().info("Subscribing to /detection/lines")
                try:
                    self.line_subscriber = self.create_subscription(
                        LineSegmentArray, "/detection/lines", self.line_detection_callback, 10)
                except NameError: # If LineSegmentArray is not defined (ImportError for dummy)
                     self.get_logger().error("Cannot subscribe to lines: LineSegmentArray message type not available. Is 'image_detector' built and sourced?")
                     self.show_lines_var.set(False) # Uncheck the box as we can't subscribe
            # else: subscriber already exists, do nothing
        else:
            if self.line_subscriber is not None:
                self.get_logger().info("Unsubscribing from /detection/lines")
                self.destroy_subscription(self.line_subscriber)
                self.line_subscriber = None
            self.detected_lines = [] # Clear data

    def line_detection_callback(self, msg):
        self.get_logger().info(f'Line detection received: {len(msg.lines)} lines')
        self.detected_lines = msg.lines

    def on_show_balls_change(self, *args):
        state = self.show_balls_var.get()
        if state:
            if self.ball_subscriber is None:
                self.get_logger().info("Subscribing to /detection/balls")
                try:
                    self.ball_subscriber = self.create_subscription(
                        BallPositionArray, "/detection/balls", self.ball_detection_callback, 10)
                except NameError: # If BallPositionArray is not defined
                    self.get_logger().error("Cannot subscribe to balls: BallPositionArray message type not available. Is 'image_detector' built and sourced?")
                    self.show_balls_var.set(False) # Uncheck the box
            # else: subscriber already exists
        else:
            if self.ball_subscriber is not None:
                self.get_logger().info("Unsubscribing from /detection/balls")
                self.destroy_subscription(self.ball_subscriber)
                self.ball_subscriber = None
            self.detected_balls = []

    def ball_detection_callback(self, msg):
        self.get_logger().info(f'Ball detection received: {len(msg.balls)} balls')
        self.detected_balls = msg.balls

    def update_gui_display(self):
        if self.current_cv_image is not None:
            display_image = self.current_cv_image.copy()

            if self.show_lines_var.get() and self.detected_lines:
                for line_segment in self.detected_lines:
                    start_point = (int(line_segment.start.x), int(line_segment.start.y))
                    end_point = (int(line_segment.end.x), int(line_segment.end.y))
                    cv2.line(display_image, start_point, end_point, (0, 255, 0), 2)

            if self.show_balls_var.get() and self.detected_balls:
                for ball_pos in self.detected_balls:
                    center = (int(ball_pos.position.x), int(ball_pos.position.y))
                    # Basic color mapping (example)
                    color_bgr = (0,0,255) # Default Red
                    if ball_pos.color.lower() == "blue":
                        color_bgr = (255,0,0)
                    elif ball_pos.color.lower() == "green":
                        color_bgr = (0,255,0)

                    cv2.circle(display_image, center, 10, color_bgr, -1)
                    cv2.putText(display_image, ball_pos.color,
                                (center[0] + 15, center[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            try:
                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                tk_image = ImageTk.PhotoImage(image=pil_image)

                self.image_label.config(image=tk_image, text="") # Clear text if image is shown
                self.image_label.image = tk_image
            except Exception as e:
                self.get_logger().error(f"Error updating GUI display: {e}")
        # else: self.image_label shows "Waiting for image..." or similar

    def update_ros_and_gui(self):
        if not self.is_running:
            return
        rclpy.spin_once(self, timeout_sec=0.01)
        self.update_gui_display()
        if self.is_running: # Check again in case on_closing was called during spin/update
            self.root.after(50, self.update_ros_and_gui)

    def on_closing(self):
        self.get_logger().info("Close button pressed. Attempting graceful shutdown.")
        self.is_running = False # Signal loop to stop

        # Explicitly destroy subscribers
        if self.image_subscriber:
            self.destroy_subscription(self.image_subscriber)
            self.image_subscriber = None
        if self.line_subscriber:
            self.destroy_subscription(self.line_subscriber)
            self.line_subscriber = None
        if self.ball_subscriber:
            self.destroy_subscription(self.ball_subscriber)
            self.ball_subscriber = None

        # Destroy node and then the Tkinter window
        if rclpy.ok() and self.is_valid():
             self.destroy_node()
        self.root.destroy()
        # rclpy.shutdown() is called in main's finally block

def main(args=None):
    rclpy.init(args=args)

    root = tk.Tk()
    app = CameraGuiApp(root) # is_running is set to True in app.__init__

    # No need to override update_ros_and_gui or on_closing like before,
    # as is_running logic is now internal to the class.
    # root.protocol("WM_DELETE_WINDOW", app.on_closing) is set in __init__.

    try:
        app.update_ros_and_gui() # Start the loop
        app.root.mainloop()
    except Exception as e:
        # This might catch errors during mainloop, e.g., if Tkinter itself has an issue
        if app.is_running : # Check if app logger is still valid
             app.get_logger().error(f"Unhandled exception in main loop: {e}")
        else:
            print(f"Unhandled exception in main loop after shutdown: {e}")
    finally:
        app.is_running = False # Ensure loop doesn't try to restart if mainloop exits unexpectedly
        if app.is_valid(): # Check if node wasn't destroyed by on_closing
            app.get_logger().info("Mainloop exited. Ensuring node is destroyed.")
            app.destroy_node()
        if rclpy.ok():
            print("Shutting down rclpy.") # Use print if logger might be gone
            rclpy.shutdown()

if __name__ == '__main__':
    main()
