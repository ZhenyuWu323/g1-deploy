import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()


config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)



frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

# print("Width:", color_intrinsics.width)
# print("Height:", color_intrinsics.height)
# print("Focal Length (fx, fy):", color_intrinsics.fx, color_intrinsics.fy)
# print("Principal Point (ppx, ppy):", color_intrinsics.ppx, color_intrinsics.ppy)
# print("Distortion Coeffs:", color_intrinsics.coeffs)

intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

fx, fy = intr.fx, intr.fy       # focal length in pixels
cx, cy = intr.ppx, intr.ppy     # principal point (optical center) in pixels

print(f"fx: {fx}, fy: {fy}")
print(f"cx: {cx}, cy: {cy}")
