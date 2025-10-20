package com.server_side_and_api_gemini.clients;

import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import com.server_side_and_api_gemini.dtos.GeminiRequest;
import com.server_side_and_api_gemini.dtos.GeminiResponse;

@Component
public class GeminiClient {

	private final WebClient webClient;
	private final String geminiApiKey;
	private static final Duration TIMEOUT = Duration.ofSeconds(45);

	public GeminiClient(@Value("${gemini.api.url}") String geminiUrl, 
						@Value("${gemini.api.key}") String geminiApiKey) {

		this.geminiApiKey = geminiApiKey;
		this.webClient = WebClient.builder().baseUrl(geminiUrl).defaultHeader("Content-Type", "application/json")
				.build();
	}

	public String getFinalDiagnosis(String prompt, String base64Image, String mimeType) {

		GeminiRequest requestBody = new GeminiRequest(prompt, base64Image, mimeType);

		try {
			GeminiResponse response = this.webClient.post()
					.uri(uriBuilder -> uriBuilder.queryParam("key", geminiApiKey).build()).bodyValue(requestBody)
					.retrieve()
					.onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
							clientResponse -> clientResponse.createException()
									.map(e -> new RuntimeException(
											"Erro " + clientResponse.statusCode() + " na API Gemini.")))
					.bodyToMono(GeminiResponse.class).timeout(TIMEOUT).block();

			if (response == null) {
				return "Erro: Resposta nula da API Gemini.";
			}

			return response.getGeneratedText();

		} catch (Exception e) {
			System.err.println("Falha ao se comunicar com a API Gemini: " + e.getMessage());
			throw new RuntimeException("Erro crítico na análise multimodal (Gemini).", e);
		}
	}
}