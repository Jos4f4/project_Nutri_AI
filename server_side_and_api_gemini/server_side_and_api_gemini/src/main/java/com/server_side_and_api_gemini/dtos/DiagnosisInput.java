package com.server_side_and_api_gemini.dtos;

import org.springframework.web.multipart.MultipartFile;

// DTO para agrupar a entrada do Controller antes de passar para o Service
public record DiagnosisInput(
    Long patientId,
    double weightKg,
    double heightCm,
    boolean isEdema,
    MultipartFile imageFile
) {}