---
title: "NVIDIA Triton Inference Server"
author:
  name: Gyoungmin.gu 
  link: https://github.com/kosaf1996
comments: true
date: 2025-04-16 04:15:00 +0900
categories: [Kubernetes]
tags: [NVIDIA, GPU, Kubernetes]
---
### Triton Inference Server


>- Triton Inference Server는 AI 추론을 간소화하는 오픈소스 추론 지원 소프트웨어
- Triton Inference Server를 통해 팀은 TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL 등 다양한 딥러닝 및 머신러닝 프레임워크에서 모든 AI 모델을 배포
- Triton은 NVIDIA GPU, x86 및 ARM CPU, 또는 AWS Inferentia를 기반으로 클라우드, 데이터 센터, 엣지 및 임베디드 디바이스 전반에서 추론을 지원
- Triton Inference Server는 실시간, 일괄 처리, 앙상블 및 오디오/비디오 스트리밍을 포함한 다양한 쿼리 유형에 최적화된 성능을 제공

[NVIDIA Triton Inference Server (공식 문서) — NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/index.html)

1. **Triton 아키텍처**

    ![nvidia-triton](/assets/img/note/nvidia-triton1.png)

    
    - Model Repository는 Triton이 추론에 사용할 모델을 저장하는 파일 시스템 기반 저장소 ( S3,  Ceph 등. )
    - 추론 요청은 HTTP, GRPC 또는 C API 를 통해 서버에 도착후 라우팅
    - Schedulter는 선택적으로 추론 요청을배치한 후 모델 유형에 해당하는 백엔드로 요청을 전달
    - 백엔드는 배치된 요청에 제공된 능력을 사용하여 추론을 수행하여 요청된 출력을 생성
    
2. **Inference Protocol & API** 
    - GRPC 프로토콜은 추론 RPC의 양방향 스트리밍 버전도 제공하여 일련의 추론 요청/응답을 GRPC 스트림을 통해 전송할 수 있도록 합니다
        
        
        >- 로드 밸런서 뒤에서 여러 Triton 서버 인스턴스가 실행되는 시스템을 가정해 보겠습니다. 일련의 추론 요청이 동일한 Triton 서버 인스턴스에 도달해야 하는 경우, GRPC 스트림은 수명 주기 동안 단일 연결을 유지하여 요청이 동일한 Triton 인스턴스로 전달되도록 합니다.
        - 네트워크 상에서 요청/응답 순서를 유지해야 하는 경우, GRPC 스트림은 서버가 클라이언트에서 보낸 순서대로 요청을 수신하도록 보장합니다.
        
    - HTTP/REST 및 GRPC 프로토콜은 서버 및 모델 상태, 메타데이터 및 통계를 확인하는 엔드포인트도 제공합니다
    
    - **HTTP**
        
        
        | Triton 서버 오류 코드 | HTTP 상태 코드 | 설명 |
        | --- | --- | --- |
        | `TRITONSERVER_ERROR_INTERNAL` | 500 | 내부 서버 오류 |
        | `TRITONSERVER_ERROR_NOT_FOUND` | 404 | 찾을 수 없음 |
        | `TRITONSERVER_ERROR_UNAVAILABLE` | 503 | 서비스를 이용할 수 없습니다 |
        | `TRITONSERVER_ERROR_UNSUPPORTED` | 501 | 구현되지 않음 |
        | `TRITONSERVER_ERROR_UNKNOWN`,
        `TRITONSERVER_ERROR_INVALID_ARG`,
        `TRITONSERVER_ERROR_ALREADY_EXISTS`,
        `TRITONSERVER_ERROR_CANCELLED` | `400` | 잘못된 요청(다른 오류에 대한 기본값) |
    
    - **GRPC**
        - **SSL/TLS**
            - `-grpc-use-ssl`
            - `-grpc-use-ssl-mutual`
            - `-grpc-server-cert`
            - `-grpc-server-key`
            - `-grpc-root-cert`
        
        - **압축**
            - `-grpc-infer-response-compression-level`
        
        - **GRPC KeepAlive**
            - `-grpc-keepalive-time`
            - `-grpc-keepalive-timeout`
            - `-grpc-keepalive-permit-without-calls`
            - `-grpc-http2-max-pings-without-data`
            - `-grpc-http2-min-recv-ping-interval-without-data`
            - `-grpc-http2-max-ping-strikes`

1. **모델 관리**
    - Triton의 모델 관리 API는 NONE, EXPLICIT 또는 POLL의 세 가지 모델 제어 모드 중 하나로 작동합니다.
    - 모델 제어 모드는 Triton이 모델 저장소의 변경 사항을 처리하는 방식과 사용 가능한 프로토콜 및 API를 결정합니다.

- **Model Control Mode { NONE }**
    - Triton은 시작 시 모델 저장소의 모든 모델을 로드하려고 시도합니다.
    - Triton이 로드할 수 없는 모델은 "사용 불가"로 표시되며 추론에 사용할 수 없습니다.
    - 모델 제어 모드는 Triton 시작 시 지정하여 선택합니다
        
        Default :  `--model-control-mode=none`
<br>
- **Model Control Mode { EXPLICIT }**
    - 시작 시 Triton은 명령줄 옵션으로 명시적으로 지정된 모델만 로드합니다
        
        `--load-model`
        
    - 시작 시 모든 모델을 로드
        
         `--load-model=*`
        
    - 시작 후 모든 모델 로드 및 언로드 작업은 모델 제어 프로토콜을 사용하여 명시적으로 시작해야합니다.
        
        `--model-control-mode=explicit`
        
    - 모델 제어 프로토콜을 사용하여 모델을 로드하고 언로드할 때 메모리가 증가하는 경우 실제 메모리 누수가 아니라 일부 시스템의 `malloc` heuristics 로 인하여 반환되지 않는 것일수 있습니다.
        
        메모리 성능을 향상 시키려면 `malloc` 대신 `tcmalloc` 또는 `jemalloc`으로 전환을 고려해야합니다.
        
    
    - **tcmalloc**
        
        ```python
        LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libtcmalloc.so.4:${LD_PRELOAD} tritonserver --model-repository=/models ...
        ```
        
    
    - **jemalloc**
        
        LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so:${LD_PRELOAD} tritonserver --model-repository=/models ...
        
    <br>
- **Model Control Mode { POLL }**
    - 모델 저자소의 변경 사항이 감지되면 Triton은 해당 변경 사항을 기반으로 필요에 따라 모델을 로드 및 언로드 합니다.
    - 이미 로드된 모델을 다시 로드하려고 할 때 어떤 이유로든 다시 로드가 실패하면 이미 로드된 모델은 변경되지 않고 로드된 상태로 유지 됩니다.
    - 다시 로드가 성공하면 새로 로드된 모델이 모델의 가용성 손실 없이 이미 로드된 모델을 대체합니다
    <br>
    - **모델 저장소 변경 Poll**
        - 모델 저장소에 변경 사항이 즉시 감지되지 않을 수 있습니다.
    `--repository-poll-secs`
       옵션을 통해 폴링 간격을 제어 할 수 있습니다. 
            
            
   > **Triton이 모델 저장소를 폴링하는 시점과 사용자가 저장소를 변경하는 시점 사이에는 동기화가 없습니다. 따라서 Triton에서 부분적이거나 불완전한 변경이 감지되어 예기치 않은 동작이 발생할 수 있습니다. 따라서 프로덕션 환경에서는 POLL 모드를 사용하지 않는 것이 좋습니다.**
            
            
  - **Poll Mode 활성화**
        - `--model-control-mode=poll` 모델 제어 모드는 Triton을 시작할 때 `--repository-poll-secs` 값이 0 이 아닌 값으로 지정하여 활성화 됩니다.
        - Triton 실행중 모델 저장소 변경이 이루어지는 경우 신중하게 수행해야 합니다.
    <br>
    - **POLL 모드에서 Triton은 다음 모델 저장소 변경**
        - 모델의 버전 정책 에 따라 사용 가능한 버전이 변경되면 기본적으로 제공되는 모델 버전이 변경될 수 있습니다.
        - 모델 디렉터리를 제거하면 저장소에서 기존 모델을 제거할 수 있습니다.
        - 모델 구성 파일 (config.pbtxt)을 변경할 수 있므ㅕ Triton은 모델을 언로드하고 다시 로드하여 새로운 모델 구성을 적용합니다.
        - 분류를 나타내는 출력에 대한 레이블을 제공하는 레이블 파일을 추가, 제거 또는 수정할 수 있으며, Triton은 모델을 언로드했다가 다시 로드하여 새 레이블을 적용합니다.
<br>
1. **Triton Inference Server Backend**
    - 백엔드는 PyTorch, TensorFlow, TensorRT 또는 ONNX 런타임과 같은 딥러닝 프레임워크를 감싸는 래퍼일 수 있습니다
    
    - **Triton Backend**
        - **TensorRT** : TensorRT 백엔드는 TensorRT 모델을 실행하는 데 사용됩니다.<br>
            
            [GitHub - triton-inference-server/tensorrt_backend: The Triton backend for TensorRT.](https://github.com/triton-inference-server/tensorrt_backend)
            <br>
        
        - **ONNX 런타임** : ONNX 런타임 백엔드는 ONNX 모델을 실행하는 데 사용됩니다.<br>
            
            [GitHub - triton-inference-server/onnxruntime_backend: The Triton backend for the ONNX Runtime.](https://github.com/triton-inference-server/onnxruntime_backend)
            <br>
        
        - **TensorFlow** : TensorFlow 백엔드는 GraphDef 및 SavedModel 형식의 TensorFlow 모델을 실행하는 데 사용됩니다<br>
            
            [GitHub - triton-inference-server/tensorflow_backend: The Triton backend for TensorFlow.](https://github.com/triton-inference-server/tensorflow_backend)
            <br>
        
        - **PyTorch** : PyTorch 백엔드는 TorchScript 및 PyTorch 2.0 형식으로 PyTorch 모델을 실행하는 데 사용됩니다.<br>
            
            [GitHub - triton-inference-server/pytorch_backend: The Triton backend for the PyTorch TorchScript models.](https://github.com/triton-inference-server/pytorch_backend)
            <br>
        
        - **OpenVINO** : OpenVINO 백엔드는 OpenVINO 모델을 실행하는데 사용됩니다.<br>
            
            [GitHub - triton-inference-server/openvino_backend: OpenVINO backend for Triton.](https://github.com/triton-inference-server/openvino_backend)
            <br>
        
        - **Python** : Python 백엔드를 사용하면 Python으로 모델 로직을 작성할 수 있습니다.
            - 백엔드를 사용하여 Python으로 작성된 전처리/후처리 코드를 실행하거나, PyTorch Python 스크립트를 TorchScript로 변환한 후 PyTorch 백엔드를 사용하는 대신 직접 실행할 수 있습니다.<br>
            
            [GitHub - triton-inference-server/python_backend: Triton backend that enables pre-process, post-processing and other logic to be implemented in Python.](https://github.com/triton-inference-server/python_backend)
            <br>
        
        - **DALI :** DALI는 딥러닝 애플리케이션의 입력 데이터 전처리를 가속화하는 최적화된 구성 요소와 실행 엔진의 집합입니다.<br>
            
            [GitHub - triton-inference-server/dali_backend: The Triton backend that allows running GPU-accelerated data pre-processing pipelines implemented in DALI's python API.](https://github.com/triton-inference-server/dali_backend)
            <br>
        
        - **FIL :** FIL 백엔드는 XGBoost 모델, LightGBM 모델, Scikit-Learn 랜덤 포레스트 모델, cuML 랜덤 포레스트 모델을 포함한 다양한 트리 기반 머신러닝 모델을 실행하는 데 사용됩니다<br>
            
            [GitHub - triton-inference-server/fil_backend: FIL backend for the Triton Inference Server](https://github.com/triton-inference-server/fil_backend)
            <br>
        
        - **TensorRT-LLM :** TensorRT-LLM 백엔드를 사용하면 Triton Server에서 TensorRT-LLM 모델을 제공할 수 있습니다.<br>
            
            [GitHub - triton-inference-server/tensorrtllm_backend: The Triton TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)
            <br>
        
        - **vLLM** : vLLM 백엔드는 vLLM 엔진에서 지원되는 모델을 실행하도록 설계되었습니다.
            - vLLM 백엔드는 python backend를 사용하여 모델을 로드하고 제공합니다.
            [GitHub - triton-inference-server/vllm_backend](https://github.com/triton-inference-server/vllm_backend)
            
        
    - **Backend**
        - 모든 모델은 백앤드와 연결되어야 하며 모델의 백엔드는 모델 구성에서 `backend`  설정을 사용하여 지정됩니다.
            
            만약 TensortRT 백엔드를 사용하는 경우 설정 값은 `tensorrt` 로 지정해야합니다.
            
        - 각 백엔드는 공유 라이브러리로 구현되어야 하며, 공유 라이브러리 이름은 `libtriton_<backend-name>.so` **형식으로 지정합니다.
        
        - 예시
            - 백엔드 B를 지정하는 모델 M 의 경우 , Triton은 다음 위치에서 백엔드 공유 라이브러리를 이 순서대로 검색합니다.
                - <모델 저장소>/M/<버전 디렉토리>/libtriton_B.so
                - <모델 저장소>/M/libtriton_B.so
                - <글로벌_백엔드_디렉토리>/B/libtriton_B.so

- **Backend API**
    - **Triton Backend는 정의된 C 인터페이스를 구현해야합니다.**
    
    - `TRITONBACKEND_Backend`
        - 백엔드 자체를 나타내며 동일한 백엔드 객체는 해당 백엔드를 사용하는 모든 모델에서 공유됩니다.
        - `TRITONBACKEND_BackendName` 과 같은 관련 API는 백엔드에 대한 정보를 가져오고 사용자 정의 상태를 백엔드와 연결하는 데 사용됩니다.
        
    - `TRITONBACKEND_Model`
        - 모델을 나타내며 Triton이 로드한 각 모델은 `TRITONBACKEND_Model` 과 연결됩니다.
        - `TRITONBACKEND_ModelBackend` PI를 사용하여 모델에서 사용하는 백엔드를 나타내는 백엔드 객체를 가져올 수 있습니다.
        - `TRITONBACKEND_ModelInitialize` 및 `TRITONBACKEND_ModelFinalize`를 구현하여 주어진 모델의 백엔드를 초기화하고 모델과 관련된 사용자 정의 상태를 관리합니다
        - `TRITONBACKEND_ModelInitialize` 및 `TRITONBACKEND_ModelFinalize`를 구현할 때 스레딩 문제를 고려해야 합니다. Triton은 특정 모델에 대해 이러한 함수를 동시에 여러 번 호출하지 않습니다. 그러나 백엔드가 여러 모델에서 사용되는 경우, Triton은 각 모델마다 다른 스레드를 사용하여 함수를 동시에 호출할 수 있습니다. 따라서 백엔드는 함수에 대한 여러 동시 호출을 처리할 수 있어야 합니다.
        
    - `TRITONBACKEND_ModelInstance`
        - Triton은 모델 구성에 지정된 instance_group 설정 에 따라 하나 이상의 모델 인스턴스를 생성합니다
        - 인스턴스는 `TRITONBACKEND_ModelInstance` 객체와 연결됩니다
        - 백엔드가 구현해야 하는 유일한 함수는 `TRITONBACKEND_ModelInstanceExecute`입니다
        - `TRITONBACKEND_ModelInstanceExecute` 함수는 Triton에서 추론 요청에 대한 추론/계산을 수행하기 위해 호출됩니다.
        - 백엔드는 `TRITONBACKEND_ModelInstanceInitialize` 및 `TRITONBACKEND_ModelInstanceFinalize` 함수도 구현하여 지정된 모델 인스턴스에 대한 백엔드를 초기화하고 모델과 관련된 사용자 정의 상태를 관리합니다

- `TRITONBACKEND_Request`
    - `TRITONBACKEND_Request` 객체는 모델에 대한 추론 요청을 나타냅니다
    - `TRITONBACKEND_ModelInstanceExecute`에서 요청 객체의 소유권을 가져오고 `TRITONBACKEND_RequestRelease`를 호출하여 각 요청을 해제해야 합니다. 단, `TRITONBACKEND_ModelInstanceExecute`가 오류를 반환하는 경우 요청 객체의 소유권은 Triton으로 반환됩니다

- `TRITONBACKEND_Response`
    - 백엔드는 응답 API를 사용하여 응답에 포함된 각 출력 텐서의 이름, 형태, 데이터 유형 및 텐서 값을 설정합니다.

- `TRITONBACKEND_BackendAttribute`
    - 속성은 Triton에서 쿼리하여 특정 기능 지원, 기본 구성 및 기타 유형의 백엔드별 동작을 알릴 수 있습니다.
    - `TRITONBACKEND_BackendSetExecutionPolicy`
    - `TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup`
        - 모델 구성에서 인스턴스 그룹을 명시적으로 정의하지 않은 경우, 이 백엔드에 대해 선호할 인스턴스 그룹의 우선 순위 목록을 정의합니다.
    - `TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading`
        - 백엔드가 동시 호출을 안전하게 처리할 수 있는지 여부를 정의합니다 `TRITONBACKEND_ModelInstanceInitialize`.
        - 모델 인스턴스를 병렬로 로드하면 인스턴스 수가 많을 때 서버 시작 시간을 개선할 수 있습니다.
        - 기본적으로 이 속성은 false로 설정되어, 명시적으로 활성화하지 않는 한 모든 백엔드에서 병렬 인스턴스 로딩이 비활성화됩니다.
        - 현재 다음 공식 백엔드는 모델 인스턴스를 병렬로 로드하는 것을 지원합니다.
            - 파이썬
            - ONNX런타임
    
    - **백엔드 초기화**
        - `TRITONBACKEND_GetBackendAttribute` 백엔드에 구현된 함수가 있는지 쿼리합니다
        - 함수는 구현 여부는 선택 사항이지만, 일반적으로 `TRITONBACKEND_BackendAttribute` 백엔드별 속성을 설정하기 위한 관련 API를 호출하는 데 사용됩니다

5. **Concurrent Model Execution**
    - Triton 아키텍처는 여러 모델 및/또는 동일 모델의 여러 인스턴스를 동일한 시스템에서 병렬로 실행할 수 있도록 합니다
    - 시스템은 GPU를 0개, 1개 또는 여러 개 사용할 수 있습니다.
    

    ![nvidia-triton](/assets/img/note/nvidia-triton2.png)

    
    - 기본적으로 동일한 모델에 대한 여러 요청이 동시에 도착하면 Triton은 다음 그림과 같이 GPU에서 한 번에 하나만 예약하여 실행을 직렬화합니다
    

    ![nvidia-triton](/assets/img/note/nvidia-triton3.png)

    
    - Triton은 각 모델에서 해당 모델의 병렬 실행 허용 횟수를 지정할 수 있는 `instance-group` 이라는 모델 구성 옵션을 지원합니다.
    - 기본적으로 Triton은 시스템에서 사용 가능한 GPU당 하나의 인스턴스를 각 모델에 제공합니다.
    - 모델 구성에서 instance_group 필드를 사용하여 모델의 실행 인스턴스 수를 변경할 수 있습니다.
    - 다음 그림은 model1이 세 개의 인스턴스를 허용하도록 구성된 경우의 모델 실행을 보여줍니다. 그림에서 볼 수 있듯이 처음 세 개의 model1 추론 요청은 즉시 병렬로 실행됩니다.
    - 네 번째 model1 추론 요청은 처음 세 개의 실행 중 하나가 완료될 때까지 기다려야 합니다.

    ![nvidia-triton](/assets/img/note/nvidia-triton4.png)

    
    - **instance-group**
        - **Multiple Model Instances**
            - 기본적으로 시스템에서 사용 가능한 각 GPU마다 모델의 실행 인스턴스가 하나씩 생성됩니다
            - 인스턴스 그룹 설정을 사용하여 모든 GPU 또는 특정 GPU에만 모델의 실행 인스턴스를 여러 개 배치할 수 있습니다
            
            ```json
              instance_group [
                {
                  count: 1
                  kind: KIND_GPU
                  gpus: [ 0 ]
                },
                {
                  count: 2
                  kind: KIND_GPU
                  gpus: [ 1, 2 ]
                }
              ]
            
            ```
            
        
        - **CPU Model Instance**
            - 인스턴스 그룹 설정은 CPU에서 모델을 실행할 수 있도록 하는 데에도 사용됩니다
            - 시스템에 GPU가 있더라도 CPU에서 모델을 실행할 수 있습니다
            
            ```json
              instance_group [
                {
                  count: 2
                  kind: KIND_CPU
                }
              ]
            
            ```
            
        
        - **Host Policy**
            - 인스턴스 그룹 설정은 호스트 정책과 연결됩니다.
            - 기본적으로 호스트 정책은 인스턴스의 장치 종류에 따라 설정됩니다
            - 예를 들어, KIND_CPU는 "cpu", KIND_MODEL은 "model", KIND_GPU는 "gpu_<gpu_id>"입니다.
            
            ```json
              instance_group [
                {
                  count: 2
                  kind: KIND_CPU
                  host_policy: "policy_0"
                }
              ]
            
            ```
            
        
        - **Rate Limiter Configuration**
            - **Resources**
                - 모델 인스턴스를 실행하는 데 필요한 리소스 집합
                - "name" 필드는 리소스를 식별하고 "count" 필드는 그룹 내 모델 인스턴스가 실행하는 데 필요한 리소스 사본 수를 나타냅니다.
                - "global" 필드는 리소스가 장치별인지 아니면 시스템 전체에서 전역적으로 공유되는지를 지정합니다.
                - 로드된 모델은 전역 및 비전역 모두 동일한 이름의 리소스를 지정할 수 없습니다
                - 리소스가 제공되지 않으면 Triton은 모델 인스턴스 실행에 리소스가 필요하지 않다고 가정하고 모델 인스턴스가 사용 가능해지는 즉시 실행을 시작합니다.
                
         - **Priority**
                - 우선순위는 모든 모델의 모든 인스턴스에 대해 우선순위를 지정하는 데 사용되는 가중치 값입니다
                - 우선순위 2인 인스턴스는 우선순위 1인 인스턴스보다 스케줄링 기회가 절반으로 주어집니다.
                
                  instance_group [
                    {
                      count: 1
                      kind: KIND_GPU
                      gpus: [ 0, 1, 2 ]
                      rate_limiter {
                        resources [
                          {
                            name: "R1"
                            count: 4
                          },
                          {
                            name: "R2"
                            global: True
                            count: 2
                          }
                        ]
                        priority: 2
                      }
                    }
                  ]
                
                
           - 위 예제는 그룹의 인스턴스가 실행을 위해 4개의 "R1" 리소스와 2개의 "R2" 리소스를 필요로 함을 지정합니다. 리소스 "R2"는 전역 리소스입니다. 또한, 인스턴스 그룹의 속도 제한 우선순위는 2입니다.
                
        
        
        >- 앙상블 모델은 Triton이 사용자 정의 모델 파이프라인을 실행하는 데 사용하는 추상화입니다
        >- 앙상블 모델과 연결된 물리적 인스턴스가 없으므로 `instance_group`필드를 지정할 수 없습니다.
        >- 앙상블을 구성하는 각 구성 모델은 `instance_group`위에서 설명한 대로 앙상블이 여러 요청을 받는 경우 해당 구성 파일에서 이를 지정하고 개별적으로 병렬 실행을 지원할 수 있습니다.
        

6. **Models And Schedulers**
    - Triton은 각 모델에 대해 독립적으로 선택할 수 있는 여러 스케줄링 및 배칭 알고리즘을 지원합니다
    - 상태 비저장(`Stateless` ) 모델 과 상태 저장(`Stateful`) 모델, 그리고 Triton이 이러한 모델 유형을 지원하기 위해 제공하는 스케줄러에 대해 설명합니다
    
    - **Stateless Models**
        - Triton의 스케줄러와 관련하여, 상태 비저장 모델은 추론 요청 간에 상태를 유지하지 않습니다
        - 상태 비저장 모델의 예로는 이미지 분류 및 객체 감지와 같은 CNN이 있습니다
        - 내부 메모리를 가진 RNN 및 유사 모델은 유지하는 상태가 추론 요청 전체에 걸쳐 있지 않는 한 상태 비저장(stateless)이 될 수 있습니다.
    
    - **Stateful Models**
        - Triton 스케줄러와 관련하여, 상태 저장 모델은 추론 요청 간에 상태를 유지합니다
        - 모델은 여러 추론 요청이 함께 발생하여 일련의 추론을 형성하고, 이 추론들은 동일한 모델 인스턴스로 라우팅되어야 모델에서 유지되는 상태가 올바르게 업데이트됩니다
        - 상태 저장 모델에는 시퀀스 배치를 이용해야 합니다.
        - 시퀀스의 모든 추론 요청이 동일한 모델 인스턴스로 라우팅되도록 하여 모델이 상태를 올바르게 유지할 수 있도록 하며  모델과 통신하여 시퀀스 시작 시점, 시퀀스 종료 시점, 시퀀스에 실행 가능한 추론 요청이 있는 시점, 그리고 시퀀스의 상관관계 ID를 알려줍니다.
        
    - **Control Inputs**
        - `ModelSequenceBatching::Control` 섹션은 시퀀스 배치가 이러한 제어에 사용해야 하는 텐서를 모델이 어떻게 노출하는지를 나타냅니다.
        
        ```json
        sequence_batching {
          control_input [
            {
              name: "START"
              control [
                {
                  kind: CONTROL_SEQUENCE_START
                  fp32_false_true: [ 0, 1 ]
                }
              ]
            },
            {
              name: "END"
              control [
                {
                  kind: CONTROL_SEQUENCE_END
                  fp32_false_true: [ 0, 1 ]
                }
              ]
            },
            {
              name: "READY"
              control [
                {
                  kind: CONTROL_SEQUENCE_READY
                  fp32_false_true: [ 0, 1 ]
                }
              ]
            },
            {
              name: "CORRID"
              control [
                {
                  kind: CONTROL_SEQUENCE_CORRID
                  data_type: TYPE_UINT64
                }
              ]
            }
          ]
        }
        
        ```
        
        - **Start**
            - 시작 입력 텐서는 구성에서 `CONTROL_SEQUENCE_START`를 사용하여 지정됩니다
            - 모델에 32비트 부동 소수점 데이터 유형을 가진 START라는 입력 텐서가 있음을 나타냅니다
            - START 텐서는 배치 크기와 같은 크기를 가진 1차원이어야 합니다
            - fp32_false_true는 시퀀스 시작이 텐서 요소 값 1로 표시되고, 시작이 아닌 것은 텐서 요소 값 0으로 표시됨을 나타냅니다.
        - **END**
            - 종료 입력 텐서는 구성에서 `CONTROL_SEQUENCE_END`를 사용하여 지정됩니다
            - 모델에 32비트 부동 소수점 데이터 유형을 가진 END라는 입력 텐서가 있음을 나타냅니다
            - fp32_false_true는 시퀀스 종료가 텐서 요소 값 1로, 종료가 아닌 경우 텐서 요소 값 0으로 나타냄을 나타냅니다.
        - **READY**
            - 준비 입력 텐서는 구성에서 `CONTROL_SEQUENCE_READY`를 사용하여 지정됩니다
            - 모델에 32비트 부동 소수점 데이터 유형을 가진 READY라는 입력 텐서가 있음을 나타냅니다
            - fp32_false_true는 시퀀스 종료가 텐서 요소 값 1로, 종료가 아닌 경우 텐서 요소 값 0으로 나타냄을 나타냅니다.
        - **CORRID**
            - 상관관계 ID 입력 텐서는 구성에서 `CONTROL_SEQUENCE_CORRID`를 사용하여 지정됩니다
            - 모델에 부호 없는 64비트 정수 데이터 유형을 가진 CORRID라는 입력 텐서가 있음을 나타냅니다
            - CORRID 텐서는 배치 크기와 동일한 크기를 가진 1차원이어야 합니다. 텐서의 각 요소는 해당 배치 슬롯에 있는 시퀀스의 상관관계 ID를 나타냅니다.

1. **Ensemble Models**
    - 앙상블 모델은 하나 이상의 모델과 그 모델 간의 입력 및 출력 텐서 연결을 파이프라인 으로 나타냅니다
    - 앙상블 모델은 "데이터 전처리 -> 추론 -> 데이터 후처리"와 같이 여러 모델이 관련된 절차를 캡슐화하는 데 사용됩니다
    - 앙상블 모델을 사용하면 중간 텐서를 전송하는 오버헤드를 피하고 Triton에 전송해야 하는 요청 수를 최소화할 수 있습니다
    - 앙상블 스케줄러는 앙상블 내 모델에서 사용하는 스케줄러와 관계없이 앙상블 모델에 반드시 사용해야 합니다
    - 앙상블 모델은 모델 구성의 `ModelEnsembling::Step` 항목으로 앙상블 내 모델 간의 데이터 흐름을 지정합니다
    - 스케줄러 는 각 단계의 출력 텐서를 수집하여 사양에 따라 다른 단계의 입력 텐서로 제공합니다.
    - 앙상블 모델은 관련 모델의 특성을 상속하므로 요청 헤더의 메타데이터는 앙상블 내 모델과 일치해야 합니다
    
    ```json
    name: "ensemble_model"
    platform: "ensemble"
    max_batch_size: 1
    input [
      {
        name: "IMAGE"
        data_type: TYPE_STRING
        dims: [ 1 ]
      }
    ]
    output [
      {
        name: "CLASSIFICATION"
        data_type: TYPE_FP32
        dims: [ 1000 ]
      },
      {
        name: "SEGMENTATION"
        data_type: TYPE_FP32
        dims: [ 3, 224, 224 ]
      }
    ]
    ensemble_scheduling {
      step [
        {
          model_name: "image_preprocess_model"
          model_version: -1
          input_map {
            key: "RAW_IMAGE"
            value: "IMAGE"
          }
          output_map {
            key: "PREPROCESSED_OUTPUT"
            value: "preprocessed_image"
          }
        },
        {
          model_name: "classification_model"
          model_version: -1
          input_map {
            key: "FORMATTED_IMAGE"
            value: "preprocessed_image"
          }
          output_map {
            key: "CLASSIFICATION_OUTPUT"
            value: "CLASSIFICATION"
          }
        },
        {
          model_name: "segmentation_model"
          model_version: -1
          input_map {
            key: "FORMATTED_IMAGE"
            value: "preprocessed_image"
          }
          output_map {
            key: "SEGMENTATION_OUTPUT"
            value: "SEGMENTATION"
          }
        }
      ]
    }
    ```
    
    - **ensemble_scheduling**
        - 앙상블 스케줄러가 사용되고 앙상블 모델이 세 가지 다른 모델로 구성됨을 나타냅니다
        - step 섹션의 각 요소는 사용할 모델과 모델의 입력 및 출력을 스케줄러가 인식하는 텐서 이름으로 매핑하는 방법을 지정합니다
            
            
            >step의 첫 번째 요소는 최신 버전의 image_preprocess_model을 사용해야 하고, 입력 RAW_IMAGE"의 내용이 "IMAGE" 텐서에서 제공되며, 출력 "PREPROCESSED_OUTPUT"의 내용이 나중에 사용할 수 있도록 "preprocessed_image" 텐서로 매핑됨을 지정합니다 
            스케줄러가 인식하는 텐서 이름은 앙상블 입력, 앙상블 출력, 그리고 input_map과 output_map의 모든 값입니다.
            
            
        
    - **ensemble  dynamic batching**
        

        ![nvidia-triton](/assets/img/note/nvidia-triton5.png)
        
        앙상블 모델에 대한 추론 요청이 수신되면 앙상블 스케줄러는 다음을 수행합니다.
        
        1. 요청의 "`IMAGE`" 텐서가 전처리 모델의 입력 "`RAW_IMAGE`"에 매핑된다는 것을 인식합니다.
        2. 앙상블 내의 모델을 확인하고 모든 입력 텐서가 준비되었으므로 전처리 모델에 내부 요청을 보냅니다.
        3. 내부 요청이 완료되었음을 인식하고 출력 텐서를 수집하여 앙상블 내에서 알려진 고유한 이름인 "`preprocessed_image`"에 콘텐츠를 매핑합니다.
        4. 새로 수집된 텐서를 앙상블 내 모델의 입력에 매핑합니다. 이 경우, "`classification_model`"과 "`segmentation_model`"의 입력이 매핑되고 준비됨으로 표시됩니다.
        5. 새로 수집된 텐서를 필요로 하는 모델을 확인하고, 입력이 준비된 모델(이 경우 분류 모델과 분할 모델)에 내부 요청을 보냅니다. 응답 순서는 개별 모델의 부하와 계산 시간에 따라 임의대로 결정됩니다.
        6. 더 이상 내부 요청을 보내지 않을 때까지 3~5단계를 반복한 다음, 앙상블 출력 이름에 매핑된 텐서로 추론 요청에 대한 응답을 보냅니다.