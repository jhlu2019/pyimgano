"""Utility helpers for PyImgAno."""

from .image_ops import (
    Compose,
    ImagePreprocessor,
    center_crop,
    load_image,
    normalize_array,
    random_horizontal_flip,
    resize_image,
    to_numpy,
)
from .image_ops_cv import (
    AugmentationPipeline,
    add_gaussian_noise,
    adjust_brightness_contrast,
    apply_augmentations,
    canny_edges,
    clahe_equalization,
    dilate,
    erode,
    find_contours,
    gaussian_blur,
    laplacian_edges,
    morphological_close,
    morphological_open,
    motion_blur,
    random_brightness_contrast,
    random_crop_and_resize,
    random_gaussian_noise,
    random_rotation,
    sharpen,
    sobel_edges,
    to_gray,
    to_gray_equalized,
)

__all__ = [
    "Compose",
    "ImagePreprocessor",
    "center_crop",
    "load_image",
    "normalize_array",
    "random_horizontal_flip",
    "resize_image",
    "to_numpy",
    "to_gray",
    "to_gray_equalized",
    "gaussian_blur",
    "sharpen",
    "add_gaussian_noise",
    "adjust_brightness_contrast",
    "motion_blur",
    "clahe_equalization",
    "canny_edges",
    "sobel_edges",
    "laplacian_edges",
    "find_contours",
    "erode",
    "dilate",
    "morphological_open",
    "morphological_close",
    "random_rotation",
    "random_crop_and_resize",
    "random_brightness_contrast",
    "random_gaussian_noise",
    "apply_augmentations",
    "AugmentationPipeline",
]

from .augmentation import (
    AUGMENTATION_REGISTRY,
    build_augmentation_pipeline,
    list_augmentations,
    register_augmentation,
    resolve_augmentation,
)
from .defect_ops import (
    adaptive_threshold,
    background_subtraction,
    defect_preprocess_pipeline,
    difference_of_gaussian,
    enhance_edges,
    gabor_filter_bank,
    local_binary_pattern,
    multi_scale_defect_map,
    normalize_illumination,
    top_hat,
    bottom_hat,
)

__all__ += [
    "AUGMENTATION_REGISTRY",
    "register_augmentation",
    "resolve_augmentation",
    "build_augmentation_pipeline",
    "list_augmentations",
]

__all__ += [
    "normalize_illumination",
    "background_subtraction",
    "adaptive_threshold",
    "top_hat",
    "bottom_hat",
    "difference_of_gaussian",
    "gabor_filter_bank",
    "enhance_edges",
    "local_binary_pattern",
    "multi_scale_defect_map",
    "defect_preprocess_pipeline",
]

# Dataset utilities
from .datasets import (
    MVTecDataset,
    BTADDataset,
    CustomDataset,
    load_dataset,
)

__all__ += [
    "MVTecDataset",
    "BTADDataset",
    "CustomDataset",
    "load_dataset",
]

# Advanced visualization utilities
from .advanced_viz import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_feature_space_tsne,
    plot_anomaly_heatmap,
    plot_multi_model_comparison,
    plot_threshold_analysis,
    create_evaluation_report,
)

__all__ += [
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_score_distribution",
    "plot_feature_space_tsne",
    "plot_anomaly_heatmap",
    "plot_multi_model_comparison",
    "plot_threshold_analysis",
    "create_evaluation_report",
]

# Model management utilities
from .model_utils import (
    save_model,
    load_model,
    save_checkpoint,
    load_checkpoint,
    get_model_info,
    export_model_config,
    profile_model,
    ModelRegistry,
    compare_models,
)

__all__ += [
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "get_model_info",
    "export_model_config",
    "profile_model",
    "ModelRegistry",
    "compare_models",
]

# Experiment tracking utilities
from .experiment_tracker import (
    Experiment,
    ExperimentTracker,
    track_experiment,
)

__all__ += [
    "Experiment",
    "ExperimentTracker",
    "track_experiment",
]

# Image I/O utilities
from .image_io import (
    ImageReader,
    ImageWriter,
    imread,
    imwrite,
    get_image_info
)

__all__ += [
    "ImageReader",
    "ImageWriter",
    "imread",
    "imwrite",
    "get_image_info",
]

# Color and quality utilities
from .color_quality import (
    ColorSpace,
    ToneMapping,
    WhiteBalance,
    QualityMetrics,
    ExposureNormalization,
    normalize_exposure,
    auto_color_correct
)

__all__ += [
    "ColorSpace",
    "ToneMapping",
    "WhiteBalance",
    "QualityMetrics",
    "ExposureNormalization",
    "normalize_exposure",
    "auto_color_correct",
]

# Geometry utilities
from .geometry import (
    CameraCalibration,
    DistortionCorrection,
    HomographyTransform,
    ROIOperations,
    CoordinateTransform,
    undistort_image,
    crop_roi,
    resize_image as geometry_resize_image
)

__all__ += [
    "CameraCalibration",
    "DistortionCorrection",
    "HomographyTransform",
    "ROIOperations",
    "CoordinateTransform",
    "undistort_image",
    "crop_roi",
    "geometry_resize_image",
]

# Acceleration utilities
from .acceleration import (
    SIMDAccelerator,
    GPUBackend,
    DLPackInterface,
    MemoryMappedOps,
    BatchProcessor as AccelBatchProcessor,
    get_optimal_backend,
    accelerate_array_op,
    check_hardware_capabilities
)

__all__ += [
    "SIMDAccelerator",
    "GPUBackend",
    "DLPackInterface",
    "MemoryMappedOps",
    "AccelBatchProcessor",
    "get_optimal_backend",
    "accelerate_array_op",
    "check_hardware_capabilities",
]

# Video utilities
from .video import (
    VideoReader,
    VideoWriter,
    VideoProcessor,
    read_video,
    write_video,
    convert_video
)

__all__ += [
    "VideoReader",
    "VideoWriter",
    "VideoProcessor",
    "read_video",
    "write_video",
    "convert_video",
]

# Data pipeline utilities
from .data_pipeline import (
    Dataset,
    ImageDataset as PipelineImageDataset,
    DataLoader as PipelineDataLoader,
    DataCache,
    CachedDataset,
    Transform,
    Compose as PipelineCompose,
    Normalize as PipelineNormalize,
    Resize as PipelineResize,
    RandomHorizontalFlip,
    ToTensor,
    BatchProcessor as DataBatchProcessor,
    create_image_dataloader,
    load_images_parallel
)

__all__ += [
    "Dataset",
    "PipelineImageDataset",
    "PipelineDataLoader",
    "DataCache",
    "CachedDataset",
    "Transform",
    "PipelineCompose",
    "PipelineNormalize",
    "PipelineResize",
    "RandomHorizontalFlip",
    "ToTensor",
    "DataBatchProcessor",
    "create_image_dataloader",
    "load_images_parallel",
]

# Deep learning integration
from .dl_integration import (
    PreProcessing,
    PostProcessing,
    ONNXWrapper,
    TorchWrapper,
    BatchInference,
    nms,
    load_onnx_model,
    batch_inference
)

__all__ += [
    "PreProcessing",
    "PostProcessing",
    "ONNXWrapper",
    "TorchWrapper",
    "BatchInference",
    "nms",
    "load_onnx_model",
    "batch_inference",
]

# Annotation utilities
from .annotations import (
    BoundingBox,
    COCOFormat,
    YOLOFormat,
    VOCFormat,
    FormatConverter,
    validate_annotations
)

__all__ += [
    "BoundingBox",
    "COCOFormat",
    "YOLOFormat",
    "VOCFormat",
    "FormatConverter",
    "validate_annotations",
]

# Visualization utilities
from .visualization import (
    BBoxVisualizer,
    MaskVisualizer,
    KeypointVisualizer,
    HeatmapVisualizer,
    TextRenderer,
    show_image,
    visualize_detections
)

__all__ += [
    "BBoxVisualizer",
    "MaskVisualizer",
    "KeypointVisualizer",
    "HeatmapVisualizer",
    "TextRenderer",
    "show_image",
    "visualize_detections",
]

# Concurrency utilities
from .concurrency import (
    ThreadPool,
    ProcessPool,
    Pipeline,
    RateLimiter,
    ResourcePool,
    AsyncTaskManager,
    ProgressTracker,
    parallel_map,
    rate_limited
)

__all__ += [
    "ThreadPool",
    "ProcessPool",
    "Pipeline",
    "RateLimiter",
    "ResourcePool",
    "AsyncTaskManager",
    "ProgressTracker",
    "parallel_map",
    "rate_limited",
]

# Security utilities
from .security import (
    ErrorCode,
    SecurityValidator,
    InputValidator,
    ResourceLimiter,
    SecureTempFile,
    SecureTempDir,
    FileHasher,
    ErrorHandler,
    validate_image_file,
    secure_load_image
)

__all__ += [
    "ErrorCode",
    "SecurityValidator",
    "InputValidator",
    "ResourceLimiter",
    "SecureTempFile",
    "SecureTempDir",
    "FileHasher",
    "ErrorHandler",
    "validate_image_file",
    "secure_load_image",
]

# CLI and engineering utilities
from .cli import (
    ArgumentParser as CLIArgumentParser,
    ConfigManager,
    Logger as CLILogger,
    ProgressBar,
    ExperimentTracker as CLIExperimentTracker,
    PluginManager,
    setup_logger,
    load_config,
    create_experiment_tracker
)

__all__ += [
    "CLIArgumentParser",
    "ConfigManager",
    "CLILogger",
    "ProgressBar",
    "CLIExperimentTracker",
    "PluginManager",
    "setup_logger",
    "load_config",
    "create_experiment_tracker",
]
