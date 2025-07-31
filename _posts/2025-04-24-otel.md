---
title: "OpenTelemetry Auto Instrumentation"
author:
  name: Gyoungmin.gu 
  link: https://github.com/kosaf1996
comments: true
date: 2025-04-23 04:15:00 +0900
categories: [Kubernetes]
tags: [OTel, Kubernetes]
---
1. **OpenTelemetry Operator Install**
    
    ```yaml
    kubectl apply -f https://github.com/open-telemetry/opentelemetry-operator/releases/latest/download/opentelemetry-operator.yaml
    ```
    
    [OpenTelemetry Operator for Kubernetes](https://opentelemetry.io/docs/kubernetes/operator/)
    
    - **Opentelemetry Operator가 관리하는 기능**
        - auto-instrumentation of the workloads using OpenTelemetry instrumentation libraries
        - Opentelemetry Collector
    <br>
2. **instrumentation.yaml**
    
    ```yaml
    apiVersion: opentelemetry.io/v1alpha1
    kind: Instrumentation
    metadata:
      name: {application}-instrumentation
      namespace: {namespace}
    spec:
      exporter:
        endpoint: http://opentelemetry-collector.opentelemetry.svc.cluster.local:4317
      propagators:
        - tracecontext
        - baggage
        - b3
      sampler:
        type: parentbased_traceidratio
        argument: "1"
      nodejs:
        image: otel/autoinstrumentation-nodejs:latest
        env:
          - name: OTEL_EXPORTER_OTLP_ENDPOINT
            value: http://opentelemetry-collector.opentelemetry.svc.cluster.local:4317
      go:
        image: otel/autoinstrumentation-go:latest
        env:
          - name: OTEL_EXPORTER_OTLP_ENDPOINT
            value: http://opentelemetry-collector.opentelemetry.svc.cluster.local:4318
    ```
    
    ```yaml
    kubectl apply -f instrumentation.yaml
    ```
    
    - `endpoint` : opentelemetry의 svc endpoint 주소
    - `propagators` : tracecontext, baggage, b3
        - 위 내용의 대해 상세 자료는 아래 링크 참조
            
            
    - `nodejs`
        - `image` : NodeJS Auto Instrumentation 에서 사용할 이미지 정보
        - `env`
            - Auto Instrumentation로 주입후 Opentelemetry-Collector로 데이터를 전송할 떄 사용하는 언어마다 사용하는 프로토콜이 상이하여 EndPoint를 `env`로 설정한다.
            
            [Injecting Auto-instrumentation](https://opentelemetry.io/docs/kubernetes/operator/automatic/)
            
    
3. **Deployment Annotation**
    
> 아래 내용은 Kustomize를 통한 Deployment Annotation 설정 예제 입니다.
    
   1. **NodeJS**
        
        ```yaml
        - patch: |-
            - op: replace
              path: /spec/template/metadata/annotations
              value:
                instrumentation.opentelemetry.io/inject-nodejs: 'true'
          target:
            kind: Deployment
        ```
        
        - `instrumentation.opentelemetry.io/inject-nodejs: 'true'` Annotaion을 추가 하며
            
            마지막의 `nodejs` 를 다른 언어로 변경하여 이용 가능하다
            
            예시) python, java 등등 
            
   2. **Go**
        
        ```yaml
        ####Security Context
        - patch: |-
            - op: add
              path: /spec/template/spec/containers/0/securityContext
              value:
                capabilities:
                  add:
                    - SYS_PTRACE
                privileged: true
                runAsUser: 0
          target:
            kind: Deployment
            
        ####OpenTelemetry 
        - patch: |-
            - op: replace
              path: /spec/template/metadata/annotations
              value:
                instrumentation.opentelemetry.io/inject-go: 'true'
                instrumentation.opentelemetry.io/otel-go-auto-target-exe: '/app/app'
          target:
            kind: Deployment
        ```
        
        - **Security Context**
            - Go 자동 계측은 eBPF를 사용하므로 높은 권한이 필요하여 Security Context에 높은 권한을 추가 합니다.
            
        - **OpenTelemetry**
            - `instrumentation.opentelemetry.io/otel-go-auto-target-exe: '/app/app'`
                
                Go OpenTelemetry 파드가 구동되며 Application 파드에서 구동되는 Application을 추적하게됩니다.
                
                Application을 추적하기위해 실행되는 경로와 Application파일의 이름을 기재합니다.
                
4. **Container 확인**
    - **NodeJS**
        
        ![nodejs](/assets/img/note/otel_nodejs.png)


        
    - **GO**
        
        ![nodejs](/assets/img/note/otel_go.png)
