CONFIG_TEMPLATE = """
name: "%MODEL_NAME%"
platform: "%BACKEND%"
max_batch_size: %MAX_BATCH%

%INPUT_SECTION%

%OUTPUT_SECTION%

%INSTANCE_GROUP_SECTION%

%DYNAMIC_BATCHING_SECTION%
"""

# --- Inputs ---
INPUT_NLP_TEMPLATE = """
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]
"""

INPUT_LM_TEMPLATE = """
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]
"""

# --- Outputs ---
OUTPUT_EMBED_TEMPLATE = """
output [
  {
    name: "%OUTPUT_NAME%"
    data_type: TYPE_FP32
    dims: [ %EMBEDDING_DIM% ]
  }
]
"""

OUTPUT_CLASSIFICATION_TEMPLATE = """
output [
  {
    name: "%OUTPUT_NAME%"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
"""

OUTPUT_LM_TEMPLATE = """
output [
  {
    name: "%OUTPUT_NAME%"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]
"""

# --- Instance groups ---
INSTANCE_GROUP_TEMPLATE = """
instance_group [
  {
    kind: %DEVICE_KIND%
    count: %INSTANCE_COUNT%
  }
]
"""

# --- Dynamic batching ---
DYNAMIC_BATCHING_TEMPLATE = """
dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 100
}
"""

EMPTY_SECTION = ""
