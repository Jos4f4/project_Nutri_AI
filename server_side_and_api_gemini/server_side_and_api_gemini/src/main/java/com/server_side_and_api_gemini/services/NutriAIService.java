package com.server_side_and_api_gemini.services; // Assumindo seu pacote final

import java.io.IOException;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.Base64;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.server_side_and_api_gemini.clients.GeminiClient;
import com.server_side_and_api_gemini.clients.XGBoostClient;
import com.server_side_and_api_gemini.dtos.DiagnosisInput;
import com.server_side_and_api_gemini.dtos.XGBoostRequest;
import com.server_side_and_api_gemini.dtos.XGBoostResponse;
import com.server_side_and_api_gemini.entities.DiagnosisRecord;
import com.server_side_and_api_gemini.entities.Patient;
import com.server_side_and_api_gemini.repositories.DiagnosisRecordRepository;
import com.server_side_and_api_gemini.repositories.PatientRepository;

@Service
public class NutriAIService {

	private final PatientRepository patientRepository;
	private final DiagnosisRecordRepository diagnosisRecordRepository;
	private final XGBoostClient xgBoostClient;
	private final GeminiClient geminiClient;

	// Construtor para Injeção de Dependência (Não iremos utilizar o @Autowired)
	public NutriAIService(PatientRepository patientRepository, DiagnosisRecordRepository diagnosisRecordRepository,
			XGBoostClient xgBoostClient, GeminiClient geminiClient) {
		this.patientRepository = patientRepository;
		this.diagnosisRecordRepository = diagnosisRecordRepository;
		this.xgBoostClient = xgBoostClient;
		this.geminiClient = geminiClient;
	}

	// O MÉTODO CENTRAL DE ORQUESTRAÇÃO
	public DiagnosisRecord runFullDiagnosis(DiagnosisInput input) throws IOException {

		Optional<Patient> patientOpt = patientRepository.findById(input.patientId());
		Patient patient = patientOpt
				.orElseThrow(() -> new RuntimeException("Paciente não encontrado. ID: " + input.patientId()));

		// 1. Processamento Inicial (Cálculo do Z-score)
		int ageMonths = calculateAgeInMonths(patient.getDateOfBirth());
		double zScoreImc = calculateZScoreImc(ageMonths, input.weightKg(), input.heightCm());

		String base64Image = Base64.getEncoder().encodeToString(input.imageFile().getBytes());
		String mimeType = input.imageFile().getContentType() != null ? input.imageFile().getContentType()
				: "image/jpeg";

		// 2. Análise Tabular (Python/XGBoost)
		XGBoostRequest xgbRequest = new XGBoostRequest(ageMonths, zScoreImc, input.isEdema() ? 1 : 0);
		XGBoostResponse xgbResponse = xgBoostClient.getPreliminaryDiagnosis(xgbRequest);
		String xgbDiagnosis = xgbResponse.getPreliminaryDiagnosis();

		// 3. Análise Multimodal (Gemini)
		String prompt = buildGeminiPrompt(patient, input.weightKg(), input.heightCm(), zScoreImc, input.isEdema(),
				xgbDiagnosis);
		String finalGeminiDiag = geminiClient.getFinalDiagnosis(prompt, base64Image, mimeType);

		// 4. Persistência (MySQL) 
		// Atualmente estamos utilizando o H2-Console, podendo ser acessado no navegador 
		DiagnosisRecord record = new DiagnosisRecord();
		record.setPatient(patient);
		record.setHeightCm(input.heightCm());
		record.setWeightKg(input.weightKg());
		record.setzScoreImc(zScoreImc);
		record.setIsEdema(input.isEdema());
		record.setXgbPrelimDiag(xgbDiagnosis);
		record.setFinalGeminiDiag(finalGeminiDiag);
		record.setImageUrl("simulated_url/diag_" + System.currentTimeMillis());

		return diagnosisRecordRepository.save(record);
	}

	// MÉTODOS AUXILIARES IMPLEMENTADOS

	/**
	 * Constrói o prompt detalhado para o Gemini, combinando todos os dados.
	 * Este prompt pode ser melhor detalhado, ou refinado
	 */
	private String buildGeminiPrompt(Patient patient, double weightKg, double heightCm, double zScoreImc,
			boolean isEdema, String xgbDiagnosis) {

		int ageMonths = calculateAgeInMonths(patient.getDateOfBirth());

		String prompt = String.format(
				"Você é um especialista em nutrição pediátrica. Analise os seguintes dados clínicos e a imagem clínica anexa. "
						+ "Forneça um diagnóstico final de desnutrição (Normal, Risco, Moderada, Grave), justificando a decisão com base nos dados e na evidência visual da imagem. "
						+ "Use os Padrões de Crescimento da OMS como referência.\n\n" +

						"--- DADOS CLÍNICOS E RESULTADOS ---\n" + "Nome: %s\n" + "Idade: %d meses\n" + "Gênero: %s\n"
						+ "Peso/Altura: %.2f kg / %.2f cm\n" + "Z-score de IMC/Idade: %.2f (Indicador crucial)\n"
						+ "Edema Presente: %s\n" + "Diagnóstico Preliminar (XGBoost): %s"
						+ "\n\n--- INSTRUÇÕES FINAIS ---\n"
						+ "Retorne APENAS um texto claro e estruturado, começando com 'Diagnóstico Final: [Categoria]' seguido pela 'Justificativa Detalhada:'.",

				patient.getName(), ageMonths, patient.getGender(), weightKg, heightCm, zScoreImc,
				isEdema ? "SIM" : "NÃO", xgbDiagnosis);
		return prompt;
	}

	/**
	 * Calcula a idade exata em meses a partir da data de nascimento.
	 */
	private int calculateAgeInMonths(LocalDate dob) {
		return (int) ChronoUnit.MONTHS.between(dob, LocalDate.now());
	}

	/**
	 * SIMULAÇÃO: Calcula o Z-score de IMC (Peso/Altura) com base em uma referência
	 * fixa.
	 */
	private double calculateZScoreImc(int ageMonths, double weightKg, double heightCm) {
		if (heightCm == 0)
			return 999.0;

		double heightM = heightCm / 100.0;
		double imc = weightKg / (heightM * heightM);

		// Referência SIMPLIFICADA (Apenas para fins de fluxo de código)
		double referenceImcMedian = 16.0 + (ageMonths / 12.0) * 0.5;
		double referenceSd = 2.0;

		return (imc - referenceImcMedian) / referenceSd;
	}
}