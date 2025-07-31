---
title: "24-08-22 Kong API Gateway 세미나 정리"
author:
  name: Gyoungmin.gu 
  link: https://github.com/kosaf1996
comments: true
date: 2024-08-22 04:15:00 +0900
categories: [Kubernetes]
tags: [Kong, 세미나, Kubernetes]
---
### Agenda

> **1. 마이크로서비스의 시작과 그 여정 with Kong API Gateway - 15mins
> 2. Use Case: Kong AI Gateway + Kong API Traffic observability with Datadog**
> **3. Shift Left testing on CI/CD by Datadog - 30mins**

1. **Kong Manger (UI 제공)**
    

    ![kong](/assets/img/note/kong1.png)

    
    - Kong API Gateway는 Kong Manger라는 UI도구를 제공
    

    ![kong](/assets/img/note/kong2.png)

    
    - Kong Mnager UI를 통해 Route, Service를 등록하여 쉽고 간결하게 API Gateway 기능을 제공
        - Example
            - Route : `https:kongapigateway/mock`
            - Service : `my-service.default.svc:80`
        - 위 Example과 같이 Kubenetes Kong API Gateway Ingress Controller를 하나 노출 하고 
        Kong API Gateway에서 Route, Service를 통하여 내부 Kubenetes의 Service로 트래픽을 보내 다른 Service들의 Ingress를 노출 하지 않음으로써 비용 절감 효과, API 중앙 집중화, 보안성 향상을 충족할 수 있음
            
            💡 `다만 트래픽 사용량의 따라 로드밸런서 성능은 고려해 볼 필요 있음`
            
    <br>
2. **DB Less 방식 제공 **
    - Kong Gateway는 경로, 서비스 및 플러그인과 같은 구성된 엔티티를 저장하기 위해 항상 데이터베이스(Postgres, 카산드라) 가 필요했으나 여러 모드를 제공함
    

    ![kong](/assets/img/note/kong3.png)

    
    - Kong Gateway는 엔티티에 대한 메모리 내 저장소만 사용하여 데이터베이스 없이 실행가능 하며 이를 `DB-less 모드`라고 한다. 
    Kong Gateway를 DB-less로 실행할 때 엔티티 구성은 선언적 구성을 사용하여 YAML 또는 JSON의 구성 파일에서 수행됩니다.
    
    - **DB 없는 모드와 선언적 구성을 결합하면 다음과 같은 여러 가지 이점**
        - 종속성 개수 감소: 사용 사례에 대한 전체 설정이 메모리에 맞는 경우 데이터베이스 설치를 관리할 필요가 없습니다.
        - CI/CD 시나리오에서의 자동화: 엔티티에 대한 구성은 Git 저장소를 통해 관리되는 단일 소스에 보관될 수 있습니다.
        - Kong Gateway에 더 많은 배포 옵션을 제공합니다.
    
   > 💡  `[**decK는**](https://docs.konghq.com/deck/) 또한 구성을 선언적으로 관리하지만, 동기화, 덤프 또는 이와 유사한 작업을 수행하려면 데이터베이스가 필요합니다. 따라서 decK는 DB 없는 모드에서 사용할 수 없습니다.`
    
    
3. **Kong AI Gateway 제공**
    - Kong API Gateway를 통해 여러 LLM 호출
        
        ![kong](/assets/img/note/kong4.png)
        
        - LLM Guard 역할을 Kong API Gateway가 해줌 (보안)
        - 보안 기능중 API를 호출 하는 과정에서 정규 표현식 형식의 정책을 통하여 특정 문자열이 포함시 전송을 하지 않도록 구성할수 있다. ( 특정 단어 필터링 기능 )
        - API 호출간 템플릿 기능 및 플러그인 제공
           
          
            [Kong Plugin Hub | Kong Docs](https://docs.konghq.com/hub)
            
        
    - 다음 릴리즈에 시맨틱 캐싱 이라는 기능 제공
        
        → 벡터 디비 와의 연계하여 시맨틱 캐싱 기능을 제공 준비중 
        
    
    >💡 **###결론**
    Kong API Gateway는 단순 API Gateway 기능에 그치지 않고 AI(ChatGPT, BARD 등…)
    `AI gateway` 기능을 제공하며 추후 `시맨틱 캐싱`이라는 기능을 추가적으로 릴리즈 할 예정으로 타 API Gateway 대비하여 AI Gateway를 제공하여 클라우드 및 온프레미스 내부 자원을 노출시키지 않고 AI Gateway 내부 Proxy를 통해 내부 자원들을 보호 할 수 있다는 점과 내부 특정 언어 (비속어) 를 필터하여 AI 문제점등을 보완할수 있을 것으로 보입니다.
    

