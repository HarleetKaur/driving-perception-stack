# Individual reflection: Harleet Kaur

## What I worked on and what I owned

During this project, I worked on developing and testing an autonomous driving perception system. I used Google Colab, VS Code, and Docker to execute different parts of the project. My work involved running model training and evaluation, checking outputs, and making the required changes when something was not working properly.

I worked with YOLO for vehicle detection, ByteTrack for object tracking, and different methods for lane detection and depth estimation. I also ran the ONNX and TensorRT benchmarking code to compare model performance. Another important part was running the ROS2 pipeline on a recorded driving video.

I was also involved in testing the results, identifying problems, preparing documentation, and updating the GitHub repository.

## What I learned

This project helped me understand how machine learning can be used in autonomous driving. Before working on it, I had limited practical experience with how different models could be combined into one system.

I learned how YOLO identifies vehicles, pedestrians, and cyclists, while ByteTrack assigns IDs to objects so they can be followed across video frames. I also understood how lane detection works and why identifying road markings can sometimes be difficult.

Another interesting part was depth estimation. I learned that a model can estimate which objects are closer or farther away without necessarily measuring their exact distance in metres.

I also learned about model performance and why accuracy, inference speed, and results for individual classes are important. Working with ROS2 and Docker gave me a better understanding of how different components can communicate and run together.

## What went wrong and how I handled it

One of the biggest challenges I faced was getting the lane detection results correctly. Initially, the green lines representing the lanes were not appearing properly in front of the car. I had to check the outputs, make changes, and run the code multiple times before getting better results.

I also faced difficulties with object tracking. Sometimes vehicles towards the sides of the road were being detected, and the tracking IDs did not appear consistent. It was confusing at first, but reviewing the output helped me understand the problem better.

Another challenge was using the required datasets and hardware. Since BDD100K was unavailable in time, KITTI was used instead. We also used a recorded driving video through ROS2 instead of CARLA because of hardware limitations.

These experiences taught me that practical projects require patience, testing, and troubleshooting.

## Limitations and what I would do next

Although the project produced useful results, there are still areas that could be improved. The tracking boxes sometimes overlap in crowded traffic, and lane detection is not always reliable in difficult road conditions. The depth estimation also gives relative distance rather than exact measurements.

If I had more time, I would test the system on different driving videos, improve the consistency of tracking IDs, and work on making lane detection more accurate. I would also like to explore real-time performance using better hardware.

## How I used AI assistance

I used AI tools during the project for technical guidance, code suggestions, understanding errors, and troubleshooting. They were particularly helpful when I was learning unfamiliar concepts or facing problems with the implementation.

I personally executed the code and commands, reviewed the outputs, checked the results, and worked on the documentation. Although AI-assisted code and guidance were part of the development process, carrying out the practical work helped me understand how the different components function.

Overall, this project was a valuable learning experience and helped me improve my confidence in working with machine learning tools.