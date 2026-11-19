# 10-minute presentation outline

## 1. Problem and Goal (0:40)

My project focuses on developing an autonomous driving perception system that can understand what is happening on the road using computer vision and machine learning.

The main goal was to detect vehicles and pedestrians, track moving objects, identify lanes, and estimate how close or far objects are. I also wanted to understand how these different components could work together in a single system.

## 2. Data and Exploratory Data Analysis (1:00)

For this project, I used the KITTI dataset through Ultralytics. Initially, BDD100K was considered, but its required data and labels were not available in time.

While exploring the dataset, I noticed that the classes were not equally represented. Cars made up around 71% of the bounding boxes, while some other classes had much fewer examples.

To simplify the training process, the original categories were combined into three classes: vehicle, pedestrian, and cyclist.

The dataset was divided into 5,485 training images, 500 validation images, and 1,496 test images.

**Show:** The EDA results chart.

## 3. Object Detection (1:30)

I worked with two YOLO models: YOLO11n as the baseline and YOLO11s as the improved model.

After running the training and evaluation notebooks, I compared their results on the test dataset.

The baseline model achieved an overall mAP50 of 0.655, while the improved model achieved 0.831.

This showed a noticeable improvement in detection performance. However, pedestrians remained one of the more challenging classes, with lower recall compared to the other categories.

One important thing I understood is that the improvement cannot be attributed to just one factor because both the model size and training duration were changed.

**Show:** The detection results and per-class performance table.

## 4. Object Tracking and Lane Detection (1:30)

For object tracking, I used YOLO with ByteTrack. The tracking system processed 1,537 video frames and generated 73 unique tracking IDs.

I faced some difficulties with tracking, especially when vehicles were close together. Some bounding boxes overlapped, and the tracking IDs were initially confusing to interpret.

Lane detection was another challenging part of the project. At first, the green lane lines were not appearing correctly in front of the car. After checking the outputs and rerunning the code, I was able to get better results.

I also compared classical lane detection with SegFormer. The classical method looks for lane markings, while SegFormer identifies road regions, so their outputs are different.

The classical method detected both lanes in 97.7% of the tested frames. This was a detection rate, not a measurement of lane accuracy.

**Show:** The tracking and lane detection screenshots.

## 5. Depth and Distance Estimation (0:50)

For depth estimation, I used a MiDaS-family model along with YOLO detections.

This part helped me understand how the system can estimate which vehicles are nearby and which are farther away.

An important limitation is that the model gives relative depth rather than actual distance in metres.

The project obtained a Spearman correlation of around 0.92 in a depth-ordering check. However, this was only a sanity check and did not prove that the distance estimates were accurate.

**Show:** The depth estimation results.

## 6. Model Optimization (0:50)

I compared three inference formats: PyTorch, ONNX Runtime, and TensorRT.

The purpose was to see how model performance changes when using different optimization methods.

TensorRT FP16 was approximately three times faster than PyTorch at the model inference step, with very little difference in detection accuracy.

This helped me understand why inference speed is important, especially for applications that need to process video quickly.

**Show:** The PyTorch, ONNX, and TensorRT benchmark table.

## 7. ROS2 Pipeline and Demonstration (1:50)

Another important part of my project was working with ROS2.

Instead of using the CARLA simulator, I used a recorded driving video because the required hardware was not available.

The ROS2 pipeline contains three main nodes: a video publisher, a perception node, and an overlay recorder.

The video publisher provides the frames, the perception node processes them for object tracking and lane detection, and the overlay recorder creates the final output video.

The pipeline was run using Docker on the CPU. It processed 512 frames and produced approximately 51 seconds of output video.

Some tracking labels overlapped in crowded traffic, and the pipeline was not running in real time.

**Show:** The architecture diagram and play approximately 40 seconds of `ros2_pipeline_demo.mp4`.

## 8. Testing and Reproducibility (0:40)

Testing was important because I wanted to check that different components were working as expected.

The project included 11 general tests and 4 ROS2-related tests.

Docker was used for the ROS2 environment, and the project also includes a requirements lock file to record the dependency versions.

These steps help make it easier to reproduce and check the project.

## 9. Challenges, Limitations, and Future Improvements (1:10)

This project was a useful learning experience, but it also came with several challenges.

The main problems I faced were getting the green lane lines to appear correctly, understanding the tracking IDs, and working within the available hardware limitations.

The system still has limitations. Tracking labels can overlap, lane detection can struggle around curves and intersections, and depth estimation does not provide exact physical distances.

If I had more time, I would improve the tracking consistency, test lane detection on more road conditions, and explore real-time processing using better hardware.

Overall, this project helped me gain practical experience with machine learning models, testing, troubleshooting, and combining different computer vision components into one system.

---

**Presentation preparation:** Rehearse the complete presentation twice with a timer. Be ready to explain the results, model choices, limitations, and comparisons in your own words.