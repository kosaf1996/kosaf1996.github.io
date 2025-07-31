---
title: "Rancher Multi-Cluster"
author:
  name: Gyoungmin.gu 
  link: https://github.com/kosaf1996
comments: true
date: 2025-04-16 04:15:00 +0900
categories: [Kubernetes]
tags: [Rancher, Kubernetes]
---
### Rancher Multi-Cluster

1. **구성**
    
    ![rancher-multicluster](/assets/img/note/rancher-multicluster.png)

    
    - **Cluster Controllers and Cluster Agents**
        - Rancher에 여러 대의 하위 쿠버네티스 클러스터가 등록되어 있으며 각 쿠버네티스 클러스터에는   Rancher의 Cluster Controller에 대한 터널을 열어주는 클러스터 Cluster Agents를 설치
    
    - **Cluster Controller**
        - 하위 클러스터의 리소스 변화를 감시
        - 클러스터 및 프로젝트에 대한 접근 제어 정책을 구성
        - 필요한 Kubernetes 엔진(RKE, EKS, GKE 등)를 호출하여 클러스터를 프로비저닝
        
    - **Cluster Agent (cattle-cluster-agent)**
        - 쿠버네티스 API 서버에 연결
        - 각 클러스터 내에서 pod, deployment 생성과 같은 워크로드를 관리
        - 각 클러스터의 정책에서 정의된 Role 및 Binding을 적용
        - 이벤트, 통계, 노드 정보 및 상태에 대해 클러스터와 Rancher 서버 간의 터널을 통해 통신

### Multi-Cluster **연동**

1. **Cluster Import**
    
    ![rancher-multicluster](/assets/img/note/rancher-multicluster1.png)


    
    - `Import Exising` 버튼을 클릭하여 Cluster Import를 진행한다.
    
    ![rancher-multicluster](/assets/img/note/rancher-multicluster2.png)


    
    - NKS Cluster를 통해 Multi Cluster 연계 예정이기 때문에 `Generic` 항목을 클릭한다.
    
    ![rancher-multicluster](/assets/img/note/rancher-multicluster3.png)
    
    - 클러스터명을 입력후 `Create`
    
2. **Multi-Cluster 연동**
    

    ![rancher-multicluster](/assets/img/note/rancher-multicluster4.png)

    
    - 생성하게 되면 Rancher URL주소를 통하여 yaml파일을 가져와 Cluster에 설치 하는 구조이다.
    - 위 이미지에서 진행한 환경은 로드밸런서에 인증서를 적용하였기 때문에 `—insecure` 옵션이 적용되지 않은 명령을 배포하였다.
    
3. **결과**
    

    ![rancher-multicluster](/assets/img/note/rancher-multicluster5.png)

    
    - `dev-cluster` 의 status 를 확인하면 `Active` 로 정상적으로 배포 및 연계되었음을 확인 가능하다.