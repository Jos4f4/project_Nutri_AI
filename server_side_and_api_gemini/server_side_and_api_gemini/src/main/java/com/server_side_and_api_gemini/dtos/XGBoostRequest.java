package com.server_side_and_api_gemini.dtos;

import com.fasterxml.jackson.annotation.JsonProperty;

//DTO para o service_ai(Spring envia)
public class XGBoostRequest {
	
	@JsonProperty("age_months")
	private int ageMonths;
	 
	@JsonProperty("z_score_imc")
	private double zScoreImc;
	
	@JsonProperty("is_edema")
	private int isEdema;

	public XGBoostRequest(int ageMonths, double zScoreImc, int isEdema) {
		this.ageMonths = ageMonths;
		this.zScoreImc = zScoreImc;
		this.isEdema = isEdema;
	}

	public int getAgeMonths() {
		return ageMonths;
	}

	public void setAgeMonths(int ageMonths) {
		this.ageMonths = ageMonths;
	}

	public double getZScoreImc() {
		return zScoreImc;
	}

	public void setZScoreImc(double zScoreImc) {
		this.zScoreImc = zScoreImc;
	}

	public int getIsEdema() {
		return isEdema;
	}

	public void setIsEdema(int isEdema) {
		this.isEdema = isEdema;
	}
}