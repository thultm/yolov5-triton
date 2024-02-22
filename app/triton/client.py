import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import sys
import os
import argparse

def get_triton_client(url: str = 'localhost:8001'):
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client

def run_inference(model_name: str, image: np.ndarray,
                  triton_client: grpcclient.InferenceServerClient):
    inputs = []
    outputs = []
    # Create the input
    inputs.append(grpcclient.InferInput('input_image', image.shape, "UINT8"))
    # Initialize the data
    inputs[0].set_data_from_numpy(image)
    # Return dims for the output
    outputs.append(grpcclient.InferRequestedOutput('result_image'))
    outputs.append(grpcclient.InferRequestedOutput('postprocess_output'))
    
    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    # Get the output
    result_image = results.as_numpy('result_image')
    postprocess_output = results.as_numpy('postprocess_output')
    return result_image, postprocess_output

def main(image_path, model_name, url):
    triton_client = get_triton_client(url)
    
    image = np.fromfile(image_path, dtype=np.uint8)
    result_image, postprocess_output = run_inference(
        model_name, image, triton_client)
    print('Postprocess output:', postprocess_output)
    print('Result image:', result_image.shape)
    
    cv2.imwrite('output.jpg', result_image)
    # Output image path
    print(f'Saved output.jpg in {os.getcwd()}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./sample/6.jpg')
    parser.add_argument('--model_name', type=str, default='yolov5m_ensemble')
    parser.add_argument('--url', type=str, default='localhost:8001')
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url)
