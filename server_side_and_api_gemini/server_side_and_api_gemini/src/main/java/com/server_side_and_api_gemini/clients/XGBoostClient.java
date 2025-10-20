package com.server_side_and_api_gemini.clients;

import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import com.server_side_and_api_gemini.dtos.XGBoostRequest;
import com.server_side_and_api_gemini.dtos.XGBoostResponse;

@Component
public class XGBoostClient {

	private final WebClient webClient;
	private static final Duration TIMEOUT = Duration.ofSeconds(10);

	public XGBoostClient(@Value("${app.services.xgboost.url}") String xgboostUrl) {
		this.webClient = WebClient.builder().baseUrl(xgboostUrl).build();
	}

	public XGBoostResponse getPreliminaryDiagnosis(XGBoostRequest request) {

		try {
			return this.webClient.post().uri("").bodyValue(request).retrieve().bodyToMono(XGBoostResponse.class)
					.timeout(TIMEOUT).block();

		} catch (Exception e) {
			System.err.println("Falha ao se comunicar com o serviço XGBoost: " + e.getMessage());
			throw new RuntimeException("Erro na análise tabular. Serviço Python indisponível.", e);
		}
	}
}