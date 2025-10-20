package com.server_side_and_api_gemini.dtos;

//DTO para o service_ai(O que o Spring RECEBE)
public class XGBoostResponse {

	private String preliminaryDiagnosis;
	private int riskCategoryId;

	public XGBoostResponse() {
		
	}

	public String getPreliminaryDiagnosis() {
		return preliminaryDiagnosis;
	}

	public void setPreliminaryDiagnosis(String preliminaryDiagnosis) {
		this.preliminaryDiagnosis = preliminaryDiagnosis;
	}

	public int getRiskCategoryId() {
		return riskCategoryId;
	}

	public void setRiskCategoryId(int riskCategoryId) {
		this.riskCategoryId = riskCategoryId;
	}
}