# Kubernetes


## Minikube no windows

### Acessar o site e baixar a versão adequada:

https://minikube.sigs.k8s.io/docs/start/?arch=%2Fwindows%2Fx86-64%2Fstable%2F.exe+download

### Renomear o arquivo para minikube.exe e adicionar ao PATH do windows

### Inicialização do Cluster Minikube

```
minikube start
```
`Cria uma máquina virtual` 

`Instala e configura os componentes básicos do Kubernetes (API Server, Controller Manager, Scheduler, etc.)`

`nó como parte do cluster e se prepara para receber comandos via kubectl`

### Verificando a Instalação e Versão

```
minikube version
minikube status
```

### Comandos Essenciais

**Inicia o cluster local**

    minikube start

**Interrompe o cluster**

    minikube stop

**Remove a instância atual do cluster**

    minikube delete

**Abre uma interface web de gerenciamento do cluster**

    minikube dashboard

**Exibe o IP da máquina virtual gerada pelo Minikube**

    minikube ip

**Exibe os logs do cluster para depuração**

    minikube logs

**Interface web para visualização do cluster**

    minikube dashboard

**Interface web para visualização do cluster**
