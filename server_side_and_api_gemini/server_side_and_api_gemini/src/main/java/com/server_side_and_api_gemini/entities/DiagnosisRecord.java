package com.server_side_and_api_gemini.entities;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "diagnosis_records")
public class DiagnosisRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id")
    private Patient patient;

    private double heightCm;
    private double weightKg;
    private double zScoreImc;
    private boolean isEdema;
    private String imageUrl;
    
    private String xgbPrelimDiag;
    
    @Lob 
    private String finalGeminiDiag;

    private Instant createdAt;

    public DiagnosisRecord() {
        this.createdAt = Instant.now();
    }

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public Patient getPatient() {
		return patient;
	}

	public void setPatient(Patient patient) {
		this.patient = patient;
	}

	public double getHeightCm() {
		return heightCm;
	}

	public void setHeightCm(double heightCm) {
		this.heightCm = heightCm;
	}

	public double getWeightKg() {
		return weightKg;
	}

	public void setWeightKg(double weightKg) {
		this.weightKg = weightKg;
	}

	public double getzScoreImc() {
		return zScoreImc;
	}

	public void setzScoreImc(double zScoreImc) {
		this.zScoreImc = zScoreImc;
	}

	public boolean isEdema() {
		return isEdema;
	}

	public void setIsEdema(boolean isEdema) {
		this.isEdema = isEdema;
	}

	public String getImageUrl() {
		return imageUrl;
	}

	public void setImageUrl(String imageUrl) {
		this.imageUrl = imageUrl;
	}

	public String getXgbPrelimDiag() {
		return xgbPrelimDiag;
	}

	public void setXgbPrelimDiag(String xgbPrelimDiag) {
		this.xgbPrelimDiag = xgbPrelimDiag;
	}

	public String getFinalGeminiDiag() {
		return finalGeminiDiag;
	}

	public void setFinalGeminiDiag(String finalGeminiDiag) {
		this.finalGeminiDiag = finalGeminiDiag;
	}

	public Instant getCreatedAt() {
		return createdAt;
	}

	public void setCreatedAt(Instant createdAt) {
		this.createdAt = createdAt;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((createdAt == null) ? 0 : createdAt.hashCode());
		result = prime * result + ((finalGeminiDiag == null) ? 0 : finalGeminiDiag.hashCode());
		long temp;
		temp = Double.doubleToLongBits(heightCm);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		result = prime * result + ((id == null) ? 0 : id.hashCode());
		result = prime * result + ((imageUrl == null) ? 0 : imageUrl.hashCode());
		result = prime * result + (isEdema ? 1231 : 1237);
		result = prime * result + ((patient == null) ? 0 : patient.hashCode());
		temp = Double.doubleToLongBits(weightKg);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		result = prime * result + ((xgbPrelimDiag == null) ? 0 : xgbPrelimDiag.hashCode());
		temp = Double.doubleToLongBits(zScoreImc);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DiagnosisRecord other = (DiagnosisRecord) obj;
		if (createdAt == null) {
			if (other.createdAt != null)
				return false;
		} else if (!createdAt.equals(other.createdAt))
			return false;
		if (finalGeminiDiag == null) {
			if (other.finalGeminiDiag != null)
				return false;
		} else if (!finalGeminiDiag.equals(other.finalGeminiDiag))
			return false;
		if (Double.doubleToLongBits(heightCm) != Double.doubleToLongBits(other.heightCm))
			return false;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		if (imageUrl == null) {
			if (other.imageUrl != null)
				return false;
		} else if (!imageUrl.equals(other.imageUrl))
			return false;
		if (isEdema != other.isEdema)
			return false;
		if (patient == null) {
			if (other.patient != null)
				return false;
		} else if (!patient.equals(other.patient))
			return false;
		if (Double.doubleToLongBits(weightKg) != Double.doubleToLongBits(other.weightKg))
			return false;
		if (xgbPrelimDiag == null) {
			if (other.xgbPrelimDiag != null)
				return false;
		} else if (!xgbPrelimDiag.equals(other.xgbPrelimDiag))
			return false;
		if (Double.doubleToLongBits(zScoreImc) != Double.doubleToLongBits(other.zScoreImc))
			return false;
		return true;
	}
}   