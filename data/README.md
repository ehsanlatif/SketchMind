# Data Directory Structure

This directory contains task-specific data for SketchMind.

## Directory Structure

Each task should have its own subdirectory organized as follows:

```
data/
└── {task_id}/                          # Task identifier (e.g., Task42_R1_1)
    ├── student_images/                 # Student sketch submissions
    │   ├── student1.jpg
    │   ├── student2.jpg
    │   └── ...
    ├── golden_standard_images/         # Reference sketches (typically 3 images per task)
    │   ├── ref1.jpg
    │   ├── ref2.jpg
    │   └── ref3.jpg
    ├── question.txt                    # Task question (optional, can use YAML in config\)
    └── rubric.txt                      # Expert designed rubric (optional, can use YAML in config\)
└── {task_id}/  
    ├── student_images/   
    │   ├── student1.jpg
    │   └── ...
    ├── golden_standard_images/   
    │   ├── ref1.jpg
    │   ├── ref2.jpg
    │   └── ref3.jpg
    ├── question.txt  
    └── rubric.txt  
```
