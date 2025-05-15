# Generative AI for Autonomous Driving

![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2505.08854-<COLOR>.svg)](https://arxiv.org/abs/2505.08854)

<p align="center">
  <img src="figures/word_cloud.png" width="100%">
</p>

We welcome anyone to contribute to this repository. Please raise issues or pull requests for any missing papers, datasets, or methods. We will update the repository regularly.

# Contents

- [Introduction](#generative-ai-for-autonomous-driving)
- [Datasets](#datasets)
  - [Single-Vehicle Perception Datasets](#single-vehicle-perception-datasets)
  - [Motion Forecasting and Cooperative Driving Datasets](#motion-forecasting-and-cooperative-driving-datasets)
  - [Simulation Based Datasets](#simulation-based-datasets)
  - [Language-Based Datasets](#language-based-datasets)
- [Methods](#methods)
  - [Image Generation Methods](#image-generation-methods)
    - [Controllable Generation](#controllable-generation)
    - [Decompositional Generation](#decompositional-generation)
  - [LiDAR Generation Methods](#lidar-generation-methods)
  - [Trajectory Generation Methods](#trajectory-generation-methods)
  - [3D Occupancy Generation Methods](#3d-occupancy-generation-methods)
  - [Video-based Scene Generation Methods](#video-based-scene-generation-methods)
  - [3D/4D Generation Methods](#3d4d-generation-methods)
  - [3D Scene Editing Methods](#3d-scene-editing-methods)
  - [LLM-based Autonomous Driving Systems](#llm-based-autonomous-driving-systems)
  - [MLLM-based Autonomous Driving Systems](#mllm-based-autonomous-driving-systems)
- [Citation](#citation)


# Datasets

## Single-Vehicle Perception Datasets

| Dataset                                                                                                     | Data Source                     | Sampling Rate | Camera Type              | LiDAR | Radar | HD Map | Annotation Type                    |
|-------------------------------------------------------------------------------------------------------------|----------------------------------|---------------|---------------------------|-------|-------|--------|------------------------------------|
| [KITTI (2012)](http://www.cvlibs.net/datasets/kitti/)                                                      | Karlsruhe, Germany              | 10 Hz         | Stereo (2 cameras)        | ✅    |       |        | 3D Bounding Boxes                 |
| [Cityscapes (2016)](https://www.cityscapes-dataset.com/)                                                   | 50 German Cities                | N/A           | Stereo (2 cameras)        |       |       |        | 2D Segmentation                   |
| [ApolloScape (2018)](http://apolloscape.auto/)                                                             | Various Cities in China         | N/A           | Stereo (2 cameras)        | ✅    |       | ✅     | Semantic Segmentation             |
| [Honda H3D (2019)](https://usa.honda-ri.com/h3d)                                                            | Bay Area, USA                   | N/A           | Frontal View (1 camera)   | ✅    |       |        | 3D Bounding Boxes                 |
| [nuScenes (2019)](https://www.nuscenes.org/)                                                               | Boston, Pittsburgh, Singapore   | 2 Hz          | Surround View (6 cameras) | ✅    | ✅    |        | 3D Bounding Boxes                 |
| [Waymo Open Dataset (2019)](https://waymo.com/open/)                                                       | Multiple US Cities              | 10 Hz         | Frontal/Side (5 cameras)  | ✅    |       | ✅     | 3D Bounding Boxes                 |
| [Argoverse (2019)](https://www.argoverse.org/)                                                             | Miami and Pittsburgh            | 10 Hz         | Surround View             | ✅    |       | ✅     | 3D Bounding Boxes                 |
| [PandaSet (2020)](https://scale.com/open-datasets/pandaset)                                                | San Francisco                   | N/A           | Surround View (7 cameras) | ✅    |       |        | 3D Bounding Boxes, Segmentation   |
| [Audi A2D2 (2020)](https://www.a2d2.audi/)                                                                  | Various Cities in Germany       | 10 Hz         | Surround View (6 cameras) | ✅    |       |        | 3D Bounding Boxes                 |
| [ONCE Dataset (2021)](https://once-for-auto-driving.github.io)                                            | Various Cities in China         | 10 Hz         | Surround View (7 cameras) | ✅    |       |        | 3D Bounding Boxes                 |


## Motion Forecasting and Cooperative Driving Datasets

| Dataset                                                                                                        | Data Source                     | Sampling Rate | Camera Type                   | LiDAR | HD Map | Annotation Type                            |
|---------------------------------------------------------------------------------------------------------------|----------------------------------|----------------|--------------------------------|--------|--------|--------------------------------------------|
| [HighD (2018)](https://levelxdata.com/highd-dataset/)                                                         | German Highways                 | N/A            | Drone (Bird's-eye View)        |        |        | Agent 2D Bounding Boxes                   |
| [INTERACTION (2019)](https://interaction-dataset.com/)                                                        | US, China, EU Intersections     | 10 Hz          | Drone and Fixed Cameras        |        | ✅     | Agent Trajectories                        |
| [PIE (2019)](https://github.com/aras62/PIE)                                                                    | Toronto, Canada                 | 30 Hz          | Frontal View (1 camera)        |        |        | Pedestrian Bounding Boxes, Intention Labels |
| [Argoverse 1 & 2 (2019, 2022)](https://www.argoverse.org/)                                                    | Miami and Pittsburgh            | 10 Hz          | Surround View                  | ✅     | ✅     | Agent Trajectories                        |
| [Lyft Level 5 (2020)](https://woven-planet.github.io/l5kit/)                                                  | Palo Alto, USA                  | 10 Hz          | Surround View                  | ✅     | ✅     | Agent 3D Bounding Boxes                   |
| [rounD (2020)](https://levelxdata.com/round-dataset/)                                                         | German Roundabouts              | N/A            | Drone (Bird's-eye View)        |        |        | Vehicle 2D Bounding Boxes                 |
| [Waymo Open Motion (2021)](https://waymo.com/open/)                                                           | Multiple US Cities              | 10 Hz          | None                           |        |        | Vehicle, Pedestrian, Cyclist Trajectories |
| [nuPlan (2021)](https://github.com/motional/nuplan-devkit)                                                    | Multiple US Cities              | 10 Hz          | Surround View                  | ✅     | ✅     | Agent 3D Bounding Boxes                   |
| [LOKI (2021)](https://thudair.baai.ac.cn/index)                                                               | Japan Intersections             | 5 Hz           | Vehicle Cameras                | ✅     | ✅     | 3D Bounding Boxes, Intention Labels       |
| [DAIR-V2X (2021)](https://thudair.baai.ac.cn/index)                                                            | China Intersections             | N/A            | Vehicle and Roadside Cameras   | ✅     |        | 3D Bounding Boxes                         |
| [exiD (2022)](https://levelxdata.com/exid-dataset/)                                                           | German Highway Exits            | N/A            | Drone (Bird's-eye View)        |        |        | Vehicle 2D Bounding Boxes                 |
| [V2X-Seq (2023)](https://github.com/AIR-Act2Act/AIR-Act2Act)                                                  | Urban Intersections             | 10 Hz          | Vehicle and Roadside Cameras   | ✅     | ✅     | 3D Agent Bounding Boxes                   |
| [V2V4Real (2023)](https://github.com/ai4r/V2V4Real)                                                            | Ohio, USA                       | 10 Hz          | Surround View                  | ✅     |        | 3D Bounding Boxes                         |
| [UniOcc (2025)](https://github.com/ai4r/UniOcc)                                                                | Various Cities in US            | 10 Hz          | Surround View                  | ✅     |        | 3D Occupancy Grids                        |

## Simulation Based Datasets

| Dataset                                                                                             | Data Source          | Camera Type         | LiDAR | HD Map | Simulation Task                   |
|-----------------------------------------------------------------------------------------------------|----------------------|----------------------|-------|--------|-----------------------------------|
| [FRIDA/FRIDA2 (2010–2012)](http://perso.lcpc.fr/tarel.jean-philippe/bdd/frida.html)                 | MATLAB               | Monocular            |       |        | Foggy Images                      |
| [SYNTHIA (2016)](http://synthia-dataset.net/)                                                       | Unity                | Multiple Views       |       |        | Rain and Fog Images               |
| [Virtual KITTI (2016 & 2019)](https://europe.naverlabs.com/research/computer-vision/virtual-kitti/) | KITTI, Unity         | Monocular/Stereo     |       |        | Real2Sim Transfer                 |
| [Playing for Benchmarks (2018)](https://playing-for-benchmarks.org/)                                | GTA-V Game Engine    | Multiple Views       | ✅    |        | Interactive Driving Simulation    |
| [Foggy Cityscapes (2018)](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)                      | Cityscapes           | Monocular            |       |        | Foggy Images                      |
| [IDDA (2020)](https://idda-dataset.github.io/)                                                      | CARLA Simulator      | Fisheye              |       |        | Semantic Segmentation             |
| [AIODrive (2021)](https://github.com/xinshuoweng/AIODrive)                                          | CARLA                | Multiple Views       | ✅    | ✅     | Long Range Point Cloud            |
| [OPV2V (2021)](https://github.com/OPV2V/OPV2V)                                                      | CARLA                | Multiple Vehicles    | ✅    |        | Cooperative Perception            |
| [Shift (2022)](https://github.com/SysCV/shift-detection-tta)                                        | CARLA                | Multiple Views       | ✅    | ✅     | Weather, Lighting Simulation      |
| [DeepAccident (2023)](https://deepaccident.github.io/)                                              | CARLA                | Multiple Views       | ✅    | ✅     | Accident Scene Simulation         |
| [WARM-3D (2024)](https://warm-3d.github.io)                                                   | CARLA                | Monocular       |      | ✅     | Sim2Real Transfer         |

## Language-Based Datasets

| Dataset | Data Source | Modality | QA Type | # QA Pairs |
|:---|:---|:---|:---|:-----------|
| [BDD-X (2018)](https://github.com/JinkyuKimUCB/BDD-X-dataset) | Dashcam Recordings | Videos (40s clips) | Ego Intention, Scene Description | 7K         |
| [DRAMA (2023)](https://usa.honda-ri.com/drama) | Japan Driving Videos | Video | Risk Object, Ego Intention, Ego Actions, Reasoning | 170K       |
| [Rank2Tell (2024)](https://usa.honda-ri.com/rank2tell) | US Driving Videos | Video | Object Importance, Ego Intention, Ego Actions, Reasoning | 300K       |
| [LingoQA (2024)](https://github.com/wayveai/LingoQA) | Driving Videos (4s clips) | Video | Scene Description, Recommended Actions, Reasoning | 419K       |
| [NuScenes-QA (2024)](https://github.com/qiantianwen/NuScenes-QA) | nuScenes | Same as nuScenes | Scene Description | 460K       |
| [DriveLM (2024)](https://github.com/OpenDriveLab/DriveLM) | nuScenes, CARLA | Same as nuScenes | Multi-step Reasoning | 360K       |
| [NuPlanQA (2025)](https://github.com/sungyeonparkk/NuPlanQA) *Not Released as of April 2025* | nuPlan | Same as nuPlan | Perception, Spatial Reasoning, Ego Intentions | 1M         |
| [NuInstruct (2024)](https://github.com/xmed-lab/NuInstruct) | nuScenes | Same as nuScenes | Instruction–Response Pairs Across 17 Task Types | 91K        |
| [doScenes (2024)](https://github.com/rossgreer/doscenes) | nuScenes | Same as nuScenes | Free-Form Driving Instructions and Scene Reference Points | 4K         |
| [MAPLM (2024)](https://github.com/LLVM-AD/MAPLM) | Chinese Cities | Image, LiDAR | Detailed Map Description (Lanes, Road, Signs) | 61K        |
| [NuScenes-MQA (2024)](https://github.com/turingmotors/NuScenes-MQA) | nuScenes | Same as nuScenes | Scene Captioning, Visual QA | 1.5M       |
| [DriveBench (2025)](https://github.com/drive-bench/toolkit) | nuScenes | Same as DriveLM | Visual QA | 20k        |
# Methods

## Image Generation Methods

### Controllable Generation

| Method | Venue | Dataset | Modeling Type | Backbone | Control Variables |
|:---|:---|:---|:---|:---|:---|
| [BEVGen](https://github.com/alexanderswerdlow/BEVGen) | IEEE RA-L'24 | nuScenes, Argoverse 2 | VQ-VAE | Transformer | BEV Map, Object Box, Text |
| [BEVControl](https://arxiv.org/abs/2308.XXXX) | arXiv'23 | nuScenes | VAE | CNN, Transformer, CLIP | BEV Sketch, Text |
| [MagicDrive](https://github.com/cure-lab/MagicDrive) | ICLR'24 | nuScenes | Diffusion, VAE | U-Net | Road Map, Object Box, Camera Pose |
| [MagicDrive3D](https://github.com/flymin/MagicDrive3D) | arXiv'24 | nuScenes | 3DGS, Diffusion, VAE | U-Net | BEV Map, Object Box, Camera Pose |
| [Drive-WM](https://github.com/BraveGroup/Drive-WM) | CVPR'24 | Driving Data | Diffusion, VAE | U-Net | Map, Text |
| [SimGen](https://github.com/metadriverse/SimGen) | NeurIPS'24 | YouTube | Diffusion, SDEdit | U-Net | BEV, Text |
| [DatasetDM](https://github.com/showlab/DatasetDM) | NeurIPS'23 | - | Diffusion, LLM, VAE | U-Net, ControlNet | Text |
| [DriveGAN](https://github.com/nv-tlabs/DriveGAN_code) | CVPR'21 | RWD | GAN, VAE | CNN, LSTM, MLP | Steering, Speed, Scene Features |
| [LightDiff](https://github.com/jinlong17/LightDiff) | CVPR'24 | nuScenes | VAE, Diffusion | U-Net | Lighting Conditions |
| Streetscapes | SIGGRAPH'24 | Google Street View | Diffusion | ControlNet | Road Map, Height Map, Camera Pose |
| [Wovogen](https://github.com/fudan-zvg/WoVoGen) | ECCV'24 | Urban Driving | Diffusion, AutoEncoder | CNN, CLIP | Text, World Volumes, Ego Actions |
| HoloDrive | arXiv'24 | nuScenes | VAE, Diffusion | U-Net, Attention | Text, 2D Layout |
| [WeatherDG](https://github.com/Jumponthemoon/WeatherDG) | arXiv'24 | Cityscapes | Diffusion, LLM | VAE, U-Net | Text |
| [UrbanArchitect](https://github.com/UrbanArchitect/UrbanArchitect) | arXiv'24 | nuScenes | Diffusion, ControlNet | VAE | Text, 3D Layout |

### Decompositional Generation

| Method | Venue | Dataset | Modeling Type | Backbone | Control Variables |
|:---|:---|:---|:---|:---|:---|
| [ChatSim](https://github.com/yifanlu0227/ChatSim) | CVPR'24 | Waymo Open Dataset | LLM, NeRF | MLP, Transformer | 3D Assets |
| [UrbanGIRAFFE](https://github.com/freemty/UrbanGIRAFFE) | ICCV'23 | KITTI-360, CLEVR-W | NeRF | MLP | Camera Pose, Panoptic Prior |
| [Sat2Scene](https://github.com/lizuoyue/sat2scene) | CVPR'24 | HoliCity, OmniCity | NeRF | MLP | Satellite Images, Layout, 3D Constraints |
| [Block-NeRF](https://github.com/freemty/UrbanGIRAFFE) | CVPR'22 | Block-NeRF Dataset | NeRF | MLP | Spatial Block Layout, 3D Constraints |
| [S-NeRF](https://github.com/fudan-zvg/S-NeRF) | CVPR'23 | nuScenes, Waymo Open Dataset | NeRF | MLP | Camera Path, 3D Constraints |
| NF-LDM | CVPR'23 | VizDoom, Replica, AVD | Diffusion, NeRF | MLP | Scene Embedding, 3D Constraints |
| [Panoptic NeRF](https://github.com/fuxiao0719/panopticnerf) | IEEE 3DIMPVT'22 | KITTI 360 | NeRF | MLP | Semantic Segmentation, 3D Constraints |
| [Neural Point Light Field](https://github.com/princeton-computational-imaging/neural-point-light-fields) | CVPR'22 | Waymo Open Dataset | NeRF | MLP | Camera Pose, 3D Constraints |
| [Neural Scene Graphs](https://github.com/princeton-computational-imaging/neural-scene-graphs) | CVPR'21 | KITTI | NeRF | MLP | Object Graph Topology, 3D Constraints |
| UniSim | CVPR'23 | PandaSet | NeRF | MLP | Agent Profile, 3D Constraints |
| CADSim | CoRL'23 | MVMC, PandaSet | Differentiable CAD Rendering | MLP | CAD Geometry, 3D Constraints |

## LiDAR Generation Methods

| Method | Venue | Dataset | Modeling Type | Backbone | Control Mechanism | Generation Type |
|:---|:---|:---|:---|:---|:---|:---|
| [LiDMs](https://github.com/hancyran/LiDAR-Diffusion) | CVPR'24 | nuScenes, KITTI-360 | Diffusion | CNN, U-Net | Multi-modal conditions | Scene Generation |
| [RangeLDM](https://github.com/WoodwindHu/RangeLDM) | ECCV'24 | KITTI-360, nuScenes | Diffusion, VAE | CNN, U-Net | Partial Point Cloud | Scene Completion, Generation |
| [LidarDM](https://github.com/vzyrianov/LidarDM) | ICRA'25 | KITTI-360, WOD | Diffusion, VAE | CNN | Semantic Map | LiDAR Simulation & Raycasting |
| [DynamicCity](https://dynamic-city.github.io/) | ICLR'25 | Occ3D, CarlaSC | Diffusion, VAE | Transformer, CNN | Layout, Trajectory, Text, Inpainting | 4D Occupancy Scene Generation |
| GenMM | arXiv'24 | BDD100K, WOD | Diffusion | U-Net, Transformer | 3D Bounding Boxes, Reference Image | Object-Level Manipulation |
| [Text2LiDAR](https://github.com/wuyang98/Text2LiDAR) | ECCV'24 | KITTI-360, nuScenes | Diffusion | Transformer | Text | Full Scene Generation |
| UltraLiDAR | CVPR'23 | PandaSet, KITTI | VQ-VAE | Transformer | Sparse Point Cloud | Scene Completion, Generation |
| [LidarGRIT](https://github.com/hamedhaghighi/LidarGRIT) | CVPR-W'24 | KITTI-360, KITTI odometry | VQ-VAE | Transformer | Unconditional | Scene Generation |
| [NeRF-LiDAR](https://github.com/fudan-zvg/NeRF-LiDAR) | CVPR'24 | nuScenes | NeRF | U-Net, MLP | Camera Poses, Multi-view Images | LiDAR Simulation |
| [LiDAR4D](https://github.com/ispc-lab/LiDAR4D) | CVPR'24 | KITTI, nuScenes | NeRF | U-Net, MLP | Camera Poses, Multi-view LiDAR Point Cloud | LiDAR Simulation |
| [DyNFL](https://github.com/prs-eth/Dynamic-LiDAR-Resimulation) | CVPR'24 | WOD | Neural SDF | MLP | LiDAR Scans, 3D Bounding Boxes | LiDAR Simulation |
| LiDARsim | CVPR'20 | LiDARsim Dataset | Physics-based Raycasting | Raycasting Engine, U-Net | 3D backgrounds, Dynamic Object Meshes | LiDAR Simulation |
| PCGen | ICRA'23 | WOD | FPA Raycasting | Raycasting Engine, MLP | Reconstructed Scenario | LiDAR Simulation |
| [LiDARGEN](https://github.com/vzyrianov/lidargen) | ECCV'22 | KITTI-360, nuScenes | Score-Based | U-Net | Sparse Point Cloud | Scene Generation |
| Yue et al. | ACM'18 | KITTI | Physics-based Raycasting | Raycasting Engine | Pre-defined In-game Scene Parameters | LiDAR Simulation |


## Trajectory Generation Methods

| Method | Venue | Dataset | Modeling Type | Backbone |
|:---|:---|:---|:---|:---|
| Kim et al. | IEEE Access'21 | Real-world Driving | CVAE | DeepConvLSTM |
| Barbié et al. | JRM'19 | Synthetic | CVAE | RNN |
| CGNS | IROS'19 | ETH/UCY, SDD | GAN | CNN |
| EvolveGraph | NeurIPS'20 | ETH/UCY, SDD, H3D | Autoregressive | GNN |
| STG-DAT | T-ITS'21 | ETH/UCY, SDD | CVAE | GNN |
| [PathGAN](https://github.com/d1024choi/pathgan_pytorch) | ETRI'21 | iSUN | GAN | CNN |
| [MID](https://github.com/Gutianpei/MID) | CVPR'22 | ETH/UCY, Stanford Drone | Diffusion | Transformer |
| [LED](https://github.com/MediaBrain-SJTU/LED) | CVPR'23 | ETH/UCY | Diffusion | Leapfrog |
| [SingularTrajectory](https://github.com/InhwanBae/SingularTrajectory) | CVPR'24 | Multiple Benchmarks | Diffusion | SVD |
| [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) | ICLR'25 | nuPlan | Diffusion | Transformer |
| [GPT-Driver](https://github.com/PointsCoder/GPT-Driver) | NeurIPS'23 | nuScenes | LLM | Transformer |
| [DriveLM](https://github.com/OpenDriveLab/DriveLM) | ECCV'24 | nuScenes | VLM | Transformer |
| [LMDrive](https://github.com/opendilab/LMDrive) | CVPR'24 | CARLA | LLM | Transformer |
| [OpenEMMA](https://github.com/taco-group/OpenEMMA) | WACV'25 | nuScenes | VLM | Transformer |
| [Desire](https://github.com/tdavchev/DESIRE) | CVPR'17 | KITTI, Stanford Drone | CVAE | RNN |
| [Trajectron](https://github.com/StanfordASL/Trajectron) | ICCV'19 | ETH/UCY | CVAE | Graph RNN |
| [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) | ECCV'20 | ETH/UCY, nuScenes | CVAE | Constrained Graph RNN |
| [Social GAN](https://github.com/agrimgupta92/sgan) | CVPR'18 | ETH/UCY | GAN | RNN |
| [SoPhie](https://github.com/coolsunxu/sophie) | CVPR'19 | ETH/UCY | GAN | Cross Attention |
| Social-BiGAT | NeurIPS'19 | ETH/UCY | Bicycle-GAN | Graph Attention Network |
| MotionDiffuser | CVPR'23 | WOMD | Diffusion | Transformer |
| SDT | OpenReview'24 | AV2 | Diffusion | Transformer |
| Westny et al. | arXiv'24 | rounD, highD | Diffusion | GNN |
| [LMTrajectory](https://github.com/InhwanBae/LMTrajectory) | CVPR'24 | ETH/UCY | LLM | Transformer |
| TrafficSim | CVPR'21 | ATG4D (private) | CVAE | GNN |
| [TrafficBots](https://github.com/zhejz/TrafficBotsV1.5) | ICRA'23 | WOMD | CVAE | MLP |
| DJINN | NeurIPS'23 | INTERACTION | Diffusion | Transformer |
| Scenario Diffusion | NeurIPS'23 | AV2 | Diffusion | UNet |
| BehaviorGPT | NeurIPS'25 | WOMD | Autoregressive | Transformer |

## 3D Occupancy Generation Methods

| Method                                                   | Venue+Year | Dataset | Modeling Type | Backbone | Control Mechanism | Generation Type | Code |
|:---------------------------------------------------------|:---|:---|:---|:---|:---|:---|:---|
| UrbanDiffusion                                           | arXiv'24 | nuScenes via Occ3D | VQ-VAE | Diffusion | BEV Layout | Static Scene | Not Released |
| DOME                                                     | arXiv'24 | nuScenes via Occ3D | VAE | DiT | Ego Trajectory | Scene and Agent Only | Not Released |
| [OccWorld](https://github.com/wzzheng/OccWorld)          | ECCV'24 | nuScenes via Occ3D | VQ-VAE | Transformer | Past Occupancy | Scene and Agent | GitHub |
| [OccSORA](https://github.com/wzzheng/OccSora) [Redacted] | arXiv'24 | nuScenes via Occ3D | VQ-VAE | DiT | Ego Trajectory, Past Occupancy | Scene and Agent | GitHub\* |
| OccLLaMA                                                 | arXiv'24 | nuScenes via Occ3D | VQ-VAE | LLaMA | Language | Scene and Agent | Not Released |
| UnO                                                      | CVPR'24 | nuScenes, Argoverse2 | Not Specified | Transformer | Past Occupancy | Semantic LiDAR | Not Released |
| [DynamicCity](https://github.com/3DTopia/DynamicCity)    | ICLR'25 | CARLA | VAE | DiT | Ego Trajectory | Scene and Agent | GitHub |


## Video-based Scene Generation Methods

> **Note:**  
> For the "Condition" column:  
> **I** = Image, **T** = Text, **E** = BEV, **B** = Bounding Boxes/Layout,  
> **D** = Depth, **C** = Camera, **M** = Maps, **A** = Driver Action,  
> **O** = Optical Flow, **J** = Trajectory, **S** = Subject, **H** = High-level instructions (Command, Goal Point).  
> Conditions in brackets are optional.

| Method | Year | Modeling | Backbone | Frames | FPS | Condition | Closed-loop | LLMs | Code |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Panacea](https://github.com/wenyuqing/panacea) | CVPR'24 | Diffusion | ControlNet | 8 | 2 | ITEBDCM |  |  | Github |
| Delphi | CoRR'24 | Diffusion | U-Net | 40 | 2 | TEBC | ✅ |  | N/A |
| [DriveDreamer](https://github.com/JeffWang987/DriveDreamer) | ECCV'24 | Diffusion | U-Net, Transformer | 32 | 12 | ITMBA |  |  | Github |
| [DriveDreamer-2](https://github.com/f1yfisher/DriveDreamer2) | ArXiv'24 | Diffusion | U-Net | 8 | 4 | T(ECI) |  | ✅ | Github |
| DriveScape | ArXiv'24 | Diffusion | U-Net | 30 | 2-10 | IMEB |  |  | N/A |
| [DriveArena](https://github.com/PJLab-ADG/DriveArena) | CoRR'24 | Diffusion, AR | U-Net | N/A | 12 | TBCM | ✅ |  | Github |
| [DriveGen](https://github.com/Hongbin98/DriveGEN) | ArXiv'24 | Diffusion | U-Net | - | - | ITB |  |  | Github |
| [DrivingDiffusion](https://github.com/shalfun/DrivingDiffusion) | ECCV'24 | Diffusion | U-Net | - | - | ITBO |  |  | Github |
| [Vista](https://github.com/OpenDriveLab/Vista) | CoRR'24 | Diffusion, AR | U-Net | 25 | 10 | I(AHJ) |  |  | Github |
| SubjectDrive | CoRR'24 | Diffusion | ControlNet | 8 | 2 | ITSB |  |  | N/A |
| GenAD | CVPR'24 | Diffusion | Transformer | 8 | 2 | ITAJ | ✅ |  | N/A |
| [DrivingWorld](https://github.com/YvanYin/DrivingWorld) | ArXiv'24 | AR | Transformer, GPT | 400 | 10 | IJ | ✅ |  | Github |
| [Doe-1](https://github.com/wzzheng/Doe) | ArXiv'24 | N/A | N/A | - | 2 | ITJ | ✅ | ✅ | Github |
| [ChatSim](https://github.com/yifanlu0227/ChatSim) | CVPR'24 | Agent | N/A | 40 | 10 | IT |  | ✅ | Github |

## 3D/4D Generation Methods

> **Note:**  
> In the "Condition" column:  
> **M** = Maps, **I** = Images/Videos, **B** = 3D Bounding Boxes/Layout, **J** = Trajectory, **T** = Text, **O** = Opacity, **C** = Camera, **A** = Driving Action.  
> \* means not presented in the original paper but supported later.  
> \dagger means reconstruction models with a generative prior.

| Method                                                                                  | Venue      | Task | Modeling Type   | Backbone | Condition | Output | Code   |
|:----------------------------------------------------------------------------------------|:-----------|:---|:----------------|:---|:----------|:---|:-------|
| InfiniCube                                                                              | ArXiv'24   | 4D Gen. | 3DGS, DiT       | 3D U-Net, ControlNet | MBJT      | Video, 3DGS | N/A    |
| [WoVoGen](https://github.com/fudan-zvg/WoVoGen)                                         | ECCV'24    | 4D Gen. | Diffusion       | 3D U-Net, Transformer | MOTA      | Video | Github |
| [DriveX](https://github.com/fudan-zvg/DriveX)                                           | ArXiv'24   | 4D Gen. | Diffusion       | U-Net | MOTA      | Video, 3DGS | Github |
| [ChatSim](https://github.com/yifanlu0227/ChatSim)                                       | CVPR'24    | 4D Gen. | NeRF, 3DGS\*    | Transformer | IT        | Video | Github |
| [MagicDrive3D](https://github.com/flymin/MagicDrive3D)                                  | CORR'24    | 4D Gen. | 3DGS            | MLP | TEBJ      | Video, 3DGS | Github |
| DreamDrive                                                                              | ArXiv'24   | 4D Gen. | 3DGS, Diffusion | MLP | IJ        | Video, 3DGS | N/A    |
| [OmniRe](https://github.com/ziyc/drivestudio)                                           | ICLR'25    | 4D Rec. | 3DGS, Graph     | N/A | I(CD)     | 3DGS, SMPL | Github |
| [4DGF](https://github.com/tobiasfshr/map4d)                                             | NeurIPS'24 | 4D Rec. | 3DGS, Graph     | N/A | IC(D)     | 3DGS | Github |
| [StreetGaussian](https://github.com/zju3dv/street_gaussians)                            | ECCV'24    | 4D Rec. | 3DGS            | N/A | ICD       | 3DGS | Github |
| DrivingGaussian                                                                         | CVPR'24    | 4D Rec. | 3DGS            | N/A, Graph | ICD       | 3DGS | N/A    |
| SGD                                                                                     | CORR'24    | 4D Rec.\dagger | 3DGS            | U-Net, ControlNet | ITCD      | 3DGS | N/A    |
| [EmerNeRF](https://github.com/NVlabs/EmerNeRF)                                          | ICLR'24    | 4D Rec. | NeRF            | MLP | ICD       | NeRF | Github |
| VastGaussian                                                                            | CVPR'24    | 3D Rec. | 3DGS            | CNN | IC        | 3DGS | N/A    |
| [CityGaussian](https://github.com/Linketic/CityGaussian)                                | ECCV'24    | 3D Rec. | 3DGS            | N/A | IC        | 3DGS | Github |
| [DNMP](https://github.com/DNMP/DNMP)                                                    | ICCV'23    | 3D Rec. | Voxel, Mesh     | MLP | ICD       | Voxel, Mesh | Github |
| [S-NeRF](https://github.com/fudan-zvg/S-NeRF)                                           | ICLR'23    | 3D Rec. | NeRF            | MLP | ICD       | NeRF | Github |
| BlockNeRF                                                                               | CVPR'22    | 3D Rec. | NeRF            | MLP | IC        | NeRF | N/A    |
| UrbanNeRF                                                                               | CVPR'22    | 3D Rec. | NeRF            | MLP | ICD       | NeRF | N/A    |
| [Julian et al.](https://github.com/princeton-computational-imaging/neural-scene-graphs) | CVPR'21    | 4D Rec. | NeRF, Graph     | MLP | IC        | NeRF | Github |
| [STORM](https://github.com/NVlabs/GaussianSTORM)         | ICLR'25    | 4D Rec. | 3DGS     | Transformer | IC        | 3DGS | Github |

## 3D Scene Editing Methods

> Here we note their supported operations and output format.

| Method | Modeling Type | Insertion | Removal | Manipulation | Camera | LiDAR | Code |
|:---|:---|:---|:---|:---|:---|:---|:---|
| UniSim | NeRF | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | N/A |
| [DrivingGaussian](https://github.com/VDIGPKU/DrivingGaussian) | 3DGS | ✔️ |  |  | ✔️ |  | Github |
| StreetGaussian | 3DGS | ✔️ | ✔️ | ✔️ | ✔️ |  | N/A |
| Generative LiDAR | Generative Inpainting | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | N/A |
| DriveEditor | SAM, Video Diffusion | ✔️ | ✔️ | ✔️ |  | ✔️ | N/A |

## LLM-based Autonomous Driving Systems

> In the condition column, QA stands for question answering, DM for decision making, ED for environment description, SU for scene understanding, and DC for driving context.

| Method | Venue | Interaction | Task | Scenario | Backbone | Strategy | Input | Output | Code |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Dilu](https://github.com/PJLab-ADG/DiLu) | ArXiv'23 | Prompting | QA | DM | GPT-4 | ReAct | ED | Action | Github |
| [Drive-Like-A-Human](https://github.com/PJLab-ADG/DriveLikeAHuman) | WACV'24 | Prompting | QA | DM | GPT-3.5 | ReAct | ED | Action | Github |
| [Driving-with-LLMs](https://github.com/wayveai/Driving-with-LLMs) | ICRA'24 | Fine-tuning | QA | SU | LLaMA-7b | None | Question | Answer | Github |
| [LaMPilot](https://github.com/PurdueDigitalTwin/LaMPilot) | CVPR'24 | Prompting | QA | SU | General LLMs | PoT | Instruction, DC | Code | Github |
| [LLaDA](https://github.com/Boyiliee/LLaDA-AV) | CVPR'24 | Prompting | QA | DM | GPT-4 | CoT | Intended Command | Action | Github |
| [GPT-driver](https://github.com/PointsCoder/GPT-Driver) | NeurIPS'23 | Fine-tuning | Planning | E2E | GPT-3.5 | CoT | Instruction, DC | Object, Action, Trajectory | Github |
| [Talk2Drive](https://github.com/PurdueDigitalTwin/Talk2Drive) | ITSC'24 | Prompting | Planning | E2E | GPT-4 | CoT | Instruction, DC | Executable Controls | Github |
| [Agent-Driver](https://github.com/USC-GVL/Agent-Driver) | COLM'24 | Prompting | Planning | E2E | GPT-3.5 | ReAct | Observation | Object, Action, Trajectory | Github |

## MLLM-based Autonomous Driving Systems

> In the condition column, VQA stands for visual question answering, SU for scene understanding, DS for driving scene, MVF for multi-view frame, and TC for transportation context.

| Method | Venue | Interaction | Task | Scenario | Backbone | Strategy | Input | Output | Code |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| HiLM-D | ArXiv'23 | Prompting | VQA | SU | MiniGPT-4 | None | Question, DS (Video) | Answer | N/A |
| [DriveLM](https://github.com/OpenDriveLab/DriveLM) | ECCV'24 | Fine-tuning | VQA | SU | BLIP-2 | CoT | Question, DS (Image) | Answer | Github |
| [Dolphins](https://vlm-driver.github.io/) | ECCV'24 | Fine-tuning | VQA | SU | OpenFlamingo | CoT | Question, DS (Video) | Answer | Github |
| [EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD) | CVPR'24 | Fine-tuning | VQA | SU | T5/T5-Large | None | Question, DS (MVF) | Answer | Github |
| [LLM-Augmented-MTR](https://github.com/SEU-zxj/LLM-Augmented-MTR) | IROS'24 | Prompting | VQA | SU | GPT-4V | CoT | Instruction, TC-Map | Context Understanding | Github |
| [LMDrive](https://github.com/opendilab/LMDrive) | CVPR'24 | Fine-tuning | Planning | E2E | LLaVA-v1.5 | CoT | Instruction, DS (MVF), LiDAR | Control Signal | Github |
| [LeGo-Drive](https://github.com/reachpranjal/lego-drive) | IROS'24 | Fine-tuning | Planning | E2E | CLIP | None | Instruction, DS (Image) | Trajectory | Github |
| [RAG-Driver](https://github.com/YuanJianhao508/RAG-Driver) | ArXiv'24 | Fine-tuning | Planning | E2E | ViT-B/32, Vicuna-1.5 | RAG | Instruction, DS (Video) | Action, Trajectory | Github |
| DriveVLM | CoRL'24 | Fine-tuning | Planning | E2E | Qwen-V | CoT | Instruction, DS (Video) | Action, Trajectory | N/A |
| EMMA | ArXiv'24 | Fine-tuning | Planning | E2E | Gemini 1.0 Nano-1 | CoT | Instruction, DS (MVF) | Object, Action, Trajectory | N/A |
| [OpenEMMA](https://github.com/taco-group/OpenEMMA) | WACV'25 | Prompting | Planning | E2E | General MLLMs | CoT | Instruction, DS (Image) | Object, Action, Trajectory | Github |


# Citation
If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@article{wang2025generative,
    title={Generative AI for Autonomous Driving: Frontiers and Opportunities},
    author={Yuping Wang and Shuo Xing and Cui Can and Renjie Li and Hongyuan Hua and Kexin Tian and Zhaobin Mo and Xiangbo Gao and Keshu Wu and Sulong Zhou and Hengxu You and Juntong Peng and Junge Zhang and Zehao Wang and Rui Song and Mingxuan Yan and Walter Zimmer and Xingcheng Zhou and Peiran Li and Zhaohan Lu and Chia-Ju Chen and Yue Huang and Ryan A. Rossi and Lichao Sun and Hongkai Yu and Zhiwen Fan and Frank Hao Yang and Yuhao Kang and Ross Greer and Chenxi Liu and Eun Hak Lee and Xuan Di and Xinyue Ye and Liu Ren and Alois Knoll and Xiaopeng Li and Shuiwang Ji and Masayoshi Tomizuka and Marco Pavone and Tianbao Yang and Jing Du and Ming-Hsuan Yang and Hua Wei and Ziran Wang and Yang Zhou and Jiachen Li and Zhengzhong Tu},
    year={2025},
    eprint={2505.08854},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```