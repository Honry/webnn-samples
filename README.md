[![lint](https://github.com/webmachinelearning/webnn-samples/workflows/lint/badge.svg)](https://github.com/webmachinelearning/webnn-samples/actions)
[![deploy](https://github.com/webmachinelearning/webnn-samples/workflows/deploy/badge.svg)](https://github.com/webmachinelearning/webnn-samples/actions)

# WebNN API Samples

* [WebNN code editor](https://webmachinelearning.github.io/webnn-samples/code/)
* [Handwritten digits classification](https://webmachinelearning.github.io/webnn-samples/lenet/)
* [Noise suppression](https://webmachinelearning.github.io/webnn-samples/nsnet2/)
* [Fast style transfer](https://webmachinelearning.github.io/webnn-samples/style_transfer/)
* [Image classification](https://webmachinelearning.github.io/webnn-samples/image_classification/)
* [Object detection](https://webmachinelearning.github.io/webnn-samples/object_detection/)
* [Semantic segmentation](https://webmachinelearning.github.io/webnn-samples/semantic_segmentation/)

### Setup & Run

```sh
>  git clone --recurse-submodules https://github.com/webmachinelearning/webnn-samples
> cd webnn-samples & npm install
```

- Create private CA key and cert, name them as `key.pem` and `cert.pem`, move to webnn-samples directory. 
(Tip: how to create CA key and cert: https://galeracluster.com/library/documentation/ssl-cert.html)

- Start https server via `node server.js`

- Open the web browser and navigate to https://127.0.0.1:8088/semantic_segmentation/
