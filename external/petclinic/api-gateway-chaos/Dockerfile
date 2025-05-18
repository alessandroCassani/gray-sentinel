FROM springcommunity/spring-petclinic-api-gateway

# Installa wget per scaricare ChaosBlade
RUN apt-get update && apt-get install -y wget tar

# Crea le directory necessarie
RUN mkdir -p /opt/chaosblade

# Scarica e installa ChaosBlade
RUN wget -O /tmp/chaosblade.tar.gz https://github.com/chaosblade-io/chaosblade/releases/download/v1.7.2/chaosblade-1.7.2-linux-amd64.tar.gz && \
    tar -xvf /tmp/chaosblade.tar.gz -C /opt && \
    ln -s /opt/chaosblade-1.7.2 /opt/chaosblade && \
    rm /tmp/chaosblade.tar.gz

# Aggiungi ChaosBlade al PATH
ENV PATH="/opt/chaosblade-1.7.2:/usr/local/bin:${PATH}"
ENV CHAOSBLADE_HOME=/opt/chaosblade-1.7.2

# Verifica l'installazione
RUN ls -la /opt/chaosblade-1.7.2 && \
    /opt/chaosblade-1.7.2/blade version