package com.server_side_and_api_gemini.dtos;

import java.util.List;

// Classe principal para a Requisição Multimodal da API do Gemini
public class GeminiRequest {

	private List<Content> contents;

	public GeminiRequest(String prompt, String base64Image, String mimeType) {

		InlineData imageData = new InlineData(mimeType, base64Image);
		Part imagePart = new Part(imageData);
		Part textPart = new Part(prompt);

		Content content = new Content(List.of(textPart, imagePart));

		this.contents = List.of(content);
	}

	public List<Content> getContents() {
		return contents;
	}

	public void setContents(List<Content> contents) {
		this.contents = contents;
	}

	// Classes Aninhadas para Requisição ---

	public static class InlineData {
		private String mimeType;
		private String data;

		public InlineData(String mimeType, String data) {
			this.mimeType = mimeType;
			this.data = data;
		}

		public String getMimeType() {
			return mimeType;
		}

		public void setMimeType(String mimeType) {
			this.mimeType = mimeType;
		}

		public String getData() {
			return data;
		}

		public void setData(String data) {
			this.data = data;
		}
	}

	public static class Part {
		private String text;
		private InlineData inlineData;

		public Part(String text) {
			this.text = text;
		}

		public Part(InlineData inlineData) {
			this.inlineData = inlineData;
		}

		public String getText() {
			return text;
		}

		public void setText(String text) {
			this.text = text;
		}

		public InlineData getInlineData() {
			return inlineData;
		}

		public void setInlineData(InlineData inlineData) {
			this.inlineData = inlineData;
		}
	}

	public static class Content {
		private List<Part> parts;

		public Content(List<Part> parts) {
			this.parts = parts;
		}

		public List<Part> getParts() {
			return parts;
		}

		public void setParts(List<Part> parts) {
			this.parts = parts;
		}
	}
}