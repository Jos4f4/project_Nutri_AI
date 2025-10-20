package com.server_side_and_api_gemini.dtos;

import java.io.Serializable;
import java.util.List;

// Classe principal para a Resposta da API Gemini
public class GeminiResponse implements Serializable {

	private List<Candidate> candidates;

	public String getGeneratedText() {
		if (candidates != null && !candidates.isEmpty()) {
			Candidate candidate = candidates.get(0);

			if (candidate.getContent() != null) {
				List<Part> parts = candidate.getContent().getParts();

				if (parts != null && !parts.isEmpty()) {
					return parts.get(0).getText();
				}
			}
		}
		return "Erro: Falha ao extrair o texto da resposta do Gemini.";
	}

	public List<Candidate> getCandidates() {
		return candidates;
	}

	public void setCandidates(List<Candidate> candidates) {
		this.candidates = candidates;
	}

	public static class Candidate implements Serializable {
		private Content content;

		public Content getContent() {
			return content;
		}

		public void setContent(Content content) {
			this.content = content;
		}
	}

	public static class Content implements Serializable {
		private List<Part> parts;

		public List<Part> getParts() {
			return parts;
		}

		public void setParts(List<Part> parts) {
			this.parts = parts;
		}
	}

	public static class Part implements Serializable {
		private String text;

		public String getText() {
			return text;
		}

		public void setText(String text) {
			this.text = text;
		}
	}
}