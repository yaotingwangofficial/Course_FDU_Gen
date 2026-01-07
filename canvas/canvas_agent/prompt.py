"""
Prompt constants for image generation agent.
All prompts used in run_v4.py are defined here for better management.
"""

# Prompt for generating initial image generation instruction
PROMPT_GENERATE_INITIAL_INSTRUCTION = """
You are an expert in crafting image generation prompts. Based on the given object–bounding box sequence, generate a natural-sounding image generation instruction within 200 words that meets the following criteria:

Object-Bounding Box Sequence (in [x, y, w, h] format):
{formatted_sequence}

Criteria:
1. Clearly specify each object and its precise location in the image (e.g., upper left, slightly right of center) based on the bounding box layout, without mentioning coordinates, bounding boxes, or technical terms.
2. Use absolute directional terms (e.g., "to the left of," "in the lower right")—avoid vague phrases like "next to" or "near."
3. Describe spatial relationships and overall composition to ensure a coherent, realistic scene.
4. The image must appear natural or photographically realistic—no design mockups, sketches, wireframes, or annotated visuals.
5. Output only the instruction text, no explanations, headings, or extra content
"""

# Prompt for judging image quality and determining if correction is needed
PROMPT_JUDGE_IMAGE_QUALITY = """
You are an expert in image quality assessment and prompt refinement. 
Analyze the generated image to determine whether its object semantics and spatial layout match the following target object–bounding box sequence:

previous instruction:
{previous_instruction}

Target object–bounding box sequence (in [x, y, w, h] format):
{formatted_sequence}

First analysis that what objects are in the given image>
Answer: <list of objects in the image>

And then evaluate the image by checking:

- Presence: Are all required objects present?
- Semantic accuracy: Does each object match its label (e.g., "chair" must be a chair, not a stool or table)?
- Spatial correctness: Is each object positioned correctly (e.g., left, right, top, bottom, center) relative to the image frame and other objects, as implied by the bounding box sequence?
- Directional relationships: Specify exact relative positions (e.g., "Object A is to the left of Object B"). Never use vague terms like "next to" or "near".
- Scale and proportion: Are object sizes visually plausible and consistent with their positions and context?
- It is not necessary to exactly match the bounding box sizes in the object–bounding box sequence; approximate spatial alignment is sufficient.  
- If corrections are needed, please consider the behavioral patterns of the image generation model and generate a new, more precise instruction based on the previous instruction.
- Your response should include [LIST_OF_OBJECTS] and [CHECK_LIST], and one of the following: `[CONVERGED]` or `[NEEDS_CORRECTION] & [NEW_INSTRUCTION]`.

Response format:

[LIST_OF_OBJECTS] <List of objects in the image>

[CHECK_LIST] (mark True or False in [] for each criterion)
- [ ] All objects are present in the image
- [ ] All objects are positioned correctly
- [ ] All objects are semantically accurate
- [ ] All objects are scaled and proportioned correctly
- [ ] The image is natural or photographically realistic
- [ ] The image meets all requirements

- If all criteria are fully satisfied:  
[CONVERGED] Image meets all requirements.

- If any discrepancy exists:  
[NEEDS_CORRECTION] Correction needed  
[NEW_INSTRUCTION] <New instruction in English>
"""

# Prompt for generating image editing instruction based on correction reason
PROMPT_GENERATE_EDIT_INSTRUCTION = """
You are an expert in crafting image generation prompts. 
Now a generated image is need to correct, and the reason for the correction is given as the following:
                    
Reason:
{reason}

Object-Bounding Box Sequence (in [x, y, w, h] format):
{formatted_sequence}


Please give an image editing instruction to correct the image based on the reason and the object-bounding box sequence.
Do not mention any terms like "bounding box" or "coordinates" in the instruction, you shold describe the image editing instruction in a natural way with position changes, 
for example, "move the chair to the left", "scale the table to the right", "rotate the chair to the front", etc.

If you think the image can be corrected by the reason, you should give the image editing instruction to correct the image.
Response format:
[INSTRUCTION] <Image editing instruction in English>

However, if you find that the currently generated image is inconsistent with the target object–bounding box sequence, 
and this cannot be corrected by revising the instructions, you should request a new image to be regenerated:
[Reject] The image is inconsistent with the target object–bounding box sequence and can only be generated from scratch again.
"""

# Prompt for detecting object bounding boxes in an image
PROMPT_DETECT_OBJECT_BBOX = """
You are an expert in object detection and spatial analysis. 
Analyze the given image and detect the bounding boxes for the following objects:

Target objects to detect:
{object_list}

For each object, provide its bounding box in normalized coordinates [x, y, w, h] format, where:
- x, y: the center coordinates of the bounding box (normalized to [0, 1])
- w, h: the width and height of the bounding box (normalized to [0, 1])

Response format (JSON-like):
{{
  "object_1": [x, y, w, h],
  "object_2": [x, y, w, h],
  ...
}}

If an object is not found in the image, set its bounding box to null.
Output only the JSON-like structure, no additional text.
"""

# Prompt for generating correction suggestions based on bbox differences
PROMPT_GENERATE_CORRECTION_SUGGESTIONS = """
You are an expert in spatial analysis and image editing. 
Compare the detected bounding boxes in the generated image with the target bounding boxes, and generate correction suggestions.

Detected bounding boxes in the image (normalized [x, y, w, h]):
{detected_bboxes}

Target bounding boxes (normalized [x, y, w, h]):
{target_bboxes}

For each object that needs correction (IOU < 0.6), provide a suggestion on how to move it.
The suggestion should be in natural language, indicating the direction and amount of movement needed.

Response format:
{{
  "object_1": "should be moved to the left a bit",
  "object_2": "should be moved to the right much",
  ...
}}

Only include objects that need correction. If no objects need correction, return an empty object {{}}.
Output only the JSON-like structure, no additional text.
"""

