#pragma once
#include "tensorflow/c/c_api.h"

#if defined(TF_MAJOR_VERSION)
#if TF_MAJOR_VERSION < 2
#error "isidorus requires TensorFlow 2.x or higher."
#endif
#endif