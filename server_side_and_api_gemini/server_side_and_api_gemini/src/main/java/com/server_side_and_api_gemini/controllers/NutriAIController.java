package com.server_side_and_api_gemini.controllers;


import java.io.IOException;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.server_side_and_api_gemini.dtos.DiagnosisInput;
import com.server_side_and_api_gemini.entities.DiagnosisRecord;
import com.server_side_and_api_gemini.services.NutriAIService;

@RestController
@RequestMapping("/api/v1/diagnose")
public class NutriAIController {

    private final NutriAIService nutriAIService;

    public NutriAIController(NutriAIService nutriAIService) {
        this.nutriAIService = nutriAIService;
    }

    @PostMapping(value = "/run-analysis", consumes = "multipart/form-data")
    public ResponseEntity<DiagnosisRecord> runAnalysis(
            @RequestParam("patientId") Long patientId,
            @RequestParam("weightKg") double weightKg,
            @RequestParam("heightCm") double heightCm,
            @RequestParam("isEdema") boolean isEdema,
            @RequestParam("imageFile") MultipartFile imageFile) {

        if (imageFile.isEmpty()) {
            return new ResponseEntity("A imagem clínica é obrigatória.", HttpStatus.BAD_REQUEST);
        }

        try {
            // Cria o DTO de entrada e chama o serviço
            DiagnosisInput input = new DiagnosisInput(patientId, weightKg, heightCm, isEdema, imageFile);
            DiagnosisRecord finalDiagnosis = nutriAIService.runFullDiagnosis(input);
            
            return ResponseEntity.ok(finalDiagnosis);

        } catch (RuntimeException | IOException e) {
            // Logar o erro detalhado aqui...
            return new ResponseEntity(e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}