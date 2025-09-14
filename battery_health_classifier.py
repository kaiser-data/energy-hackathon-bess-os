#!/usr/bin/env python3
"""
Professional Battery Health Classification System

Industry-standard SOH (State of Health) classifications with professional terminology
used in commercial BESS operations and battery management systems.
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    """Professional battery health status classifications"""
    EXCELLENT = "excellent"          # 99-100% SOH
    OPTIMAL = "optimal"             # 98-99% SOH
    NOMINAL = "nominal"             # 95-98% SOH
    DEGRADED = "degraded"           # 90-95% SOH
    COMPROMISED = "compromised"     # 85-90% SOH
    CRITICAL = "critical"           # 80-85% SOH
    END_OF_LIFE = "end_of_life"    # <80% SOH

@dataclass
class HealthClassification:
    """Professional health assessment result"""
    status: HealthStatus
    soh_percentage: float
    capacity_retention: float
    degradation_category: str
    maintenance_action: str
    operational_impact: str
    replacement_timeline: str
    risk_assessment: str

class BatteryHealthClassifier:
    """Professional battery health classification system"""

    def __init__(self):
        # Industry-standard SOH thresholds (%)
        self.health_thresholds = {
            HealthStatus.EXCELLENT: (99.0, 100.0),
            HealthStatus.OPTIMAL: (98.0, 99.0),
            HealthStatus.NOMINAL: (95.0, 98.0),
            HealthStatus.DEGRADED: (90.0, 95.0),
            HealthStatus.COMPROMISED: (85.0, 90.0),
            HealthStatus.CRITICAL: (80.0, 85.0),
            HealthStatus.END_OF_LIFE: (0.0, 80.0)
        }

        # Professional terminology mapping
        self.professional_terms = {
            HealthStatus.EXCELLENT: {
                "description": "Peak Performance Condition",
                "technical_term": "Pristine State",
                "capacity_status": "Full Rated Capacity",
                "degradation_category": "Minimal Degradation",
                "maintenance_action": "Preventive Monitoring",
                "operational_impact": "No Performance Impact",
                "replacement_timeline": "> 8 years remaining",
                "risk_assessment": "Negligible Risk"
            },
            HealthStatus.OPTIMAL: {
                "description": "Excellent Operating Condition",
                "technical_term": "Near-Pristine State",
                "capacity_status": "Rated Capacity Maintained",
                "degradation_category": "Early-Stage Degradation",
                "maintenance_action": "Standard Monitoring",
                "operational_impact": "Negligible Impact",
                "replacement_timeline": "6-8 years remaining",
                "risk_assessment": "Very Low Risk"
            },
            HealthStatus.NOMINAL: {
                "description": "Normal Operating Condition",
                "technical_term": "Serviceable State",
                "capacity_status": "Acceptable Capacity Retention",
                "degradation_category": "Normal Aging Process",
                "maintenance_action": "Routine Inspection",
                "operational_impact": "Minimal Impact",
                "replacement_timeline": "4-6 years remaining",
                "risk_assessment": "Low Risk"
            },
            HealthStatus.DEGRADED: {
                "description": "Reduced Performance Condition",
                "technical_term": "Aged State",
                "capacity_status": "Reduced Capacity",
                "degradation_category": "Accelerated Degradation",
                "maintenance_action": "Enhanced Monitoring",
                "operational_impact": "Moderate Performance Reduction",
                "replacement_timeline": "2-4 years remaining",
                "risk_assessment": "Medium Risk"
            },
            HealthStatus.COMPROMISED: {
                "description": "Impaired Operating Condition",
                "technical_term": "Degraded State",
                "capacity_status": "Significantly Reduced Capacity",
                "degradation_category": "Advanced Degradation",
                "maintenance_action": "Intensive Monitoring",
                "operational_impact": "Notable Performance Impact",
                "replacement_timeline": "1-2 years remaining",
                "risk_assessment": "High Risk"
            },
            HealthStatus.CRITICAL: {
                "description": "Severely Compromised Condition",
                "technical_term": "Critical State",
                "capacity_status": "Critically Low Capacity",
                "degradation_category": "Severe Degradation",
                "maintenance_action": "Immediate Assessment Required",
                "operational_impact": "Significant Performance Degradation",
                "replacement_timeline": "6-12 months remaining",
                "risk_assessment": "Critical Risk"
            },
            HealthStatus.END_OF_LIFE: {
                "description": "End-of-Service-Life Condition",
                "technical_term": "EOL State",
                "capacity_status": "Below Operational Threshold",
                "degradation_category": "End-Stage Degradation",
                "maintenance_action": "Replacement Planning",
                "operational_impact": "Severe Performance Limitations",
                "replacement_timeline": "Immediate replacement required",
                "risk_assessment": "Unacceptable Risk"
            }
        }

        # Professional pack-level classifications
        self.pack_classifications = {
            "uniform_excellent": "Homogeneous High-Performance Pack",
            "uniform_optimal": "Balanced Optimal Performance Pack",
            "uniform_nominal": "Standard Performance Pack",
            "mixed_performance": "Heterogeneous Performance Pack",
            "imbalanced": "Cell Imbalance Detected",
            "degradation_spread": "Non-Uniform Degradation Pattern",
            "critical_imbalance": "Critical Cell Variance",
            "replacement_candidate": "Pack Replacement Candidate"
        }

    def classify_cell_health(self, soh_percentage: float,
                           additional_metrics: Optional[Dict] = None) -> HealthClassification:
        """Classify individual cell health with professional terminology"""

        # Determine health status
        status = self._determine_health_status(soh_percentage)

        # Get professional terms
        terms = self.professional_terms[status]

        # Calculate capacity retention
        capacity_retention = soh_percentage / 100.0

        # Create professional classification
        classification = HealthClassification(
            status=status,
            soh_percentage=soh_percentage,
            capacity_retention=capacity_retention,
            degradation_category=terms["degradation_category"],
            maintenance_action=terms["maintenance_action"],
            operational_impact=terms["operational_impact"],
            replacement_timeline=terms["replacement_timeline"],
            risk_assessment=terms["risk_assessment"]
        )

        return classification

    def classify_pack_health(self, cell_soh_values: List[float]) -> Dict:
        """Classify pack-level health with professional analysis"""

        if not cell_soh_values:
            return {"error": "No SOH data provided"}

        # Calculate pack statistics
        min_soh = min(cell_soh_values)
        max_soh = max(cell_soh_values)
        avg_soh = sum(cell_soh_values) / len(cell_soh_values)
        soh_std = self._calculate_std(cell_soh_values, avg_soh)
        soh_range = max_soh - min_soh

        # Classify individual cells
        cell_classifications = [self.classify_cell_health(soh) for soh in cell_soh_values]

        # Count cells by status
        status_counts = {}
        for classification in cell_classifications:
            status = classification.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Determine pack classification
        pack_class = self._classify_pack_uniformity(cell_soh_values, soh_std, soh_range)

        # Professional assessment
        worst_cell_soh = min(cell_soh_values)
        best_cell_soh = max(cell_soh_values)

        # Pack-level professional metrics
        pack_analysis = {
            "pack_id": "Pack Analysis",
            "overall_health_status": self._determine_health_status(avg_soh),
            "pack_classification": pack_class,

            # Statistical Analysis
            "statistical_metrics": {
                "mean_soh": round(avg_soh, 2),
                "soh_standard_deviation": round(soh_std, 3),
                "soh_coefficient_of_variation": round((soh_std / avg_soh) * 100, 2),
                "soh_range": round(soh_range, 3),
                "uniformity_index": round(100 - (soh_range / avg_soh * 100), 1)
            },

            # Cell Distribution Analysis
            "cell_distribution": {
                "total_cells": len(cell_soh_values),
                "status_distribution": {status.value: count for status, count in status_counts.items()},
                "weakest_cell_soh": round(worst_cell_soh, 2),
                "strongest_cell_soh": round(best_cell_soh, 2),
                "performance_spread": round(best_cell_soh - worst_cell_soh, 3)
            },

            # Professional Assessment
            "professional_assessment": {
                "capacity_fade_analysis": self._assess_capacity_fade(avg_soh),
                "balancing_requirement": self._assess_balancing_needs(soh_std, soh_range),
                "maintenance_priority": self._determine_maintenance_priority(status_counts),
                "operational_constraints": self._assess_operational_constraints(worst_cell_soh),
                "replacement_strategy": self._recommend_replacement_strategy(status_counts, soh_range)
            },

            # Risk Analysis
            "risk_analysis": {
                "thermal_runaway_risk": self._assess_thermal_risk(status_counts),
                "performance_degradation_rate": self._estimate_degradation_rate(avg_soh),
                "system_reliability_impact": self._assess_reliability_impact(worst_cell_soh, soh_std),
                "safety_classification": self._classify_safety_status(worst_cell_soh, soh_range)
            }
        }

        return pack_analysis

    def _determine_health_status(self, soh_percentage: float) -> HealthStatus:
        """Determine health status from SOH percentage"""
        for status, (min_soh, max_soh) in self.health_thresholds.items():
            if min_soh <= soh_percentage <= max_soh:
                return status
        return HealthStatus.END_OF_LIFE

    def _classify_pack_uniformity(self, soh_values: List[float],
                                 std_dev: float, soh_range: float) -> str:
        """Classify pack uniformity with professional terms"""

        avg_soh = sum(soh_values) / len(soh_values)
        cv = (std_dev / avg_soh) * 100  # Coefficient of Variation

        if avg_soh >= 99 and cv < 0.5:
            return self.pack_classifications["uniform_excellent"]
        elif avg_soh >= 98 and cv < 1.0:
            return self.pack_classifications["uniform_optimal"]
        elif avg_soh >= 95 and cv < 2.0:
            return self.pack_classifications["uniform_nominal"]
        elif cv > 5.0:
            return self.pack_classifications["critical_imbalance"]
        elif soh_range > 10.0:
            return self.pack_classifications["degradation_spread"]
        elif cv > 3.0:
            return self.pack_classifications["imbalanced"]
        elif avg_soh < 85:
            return self.pack_classifications["replacement_candidate"]
        else:
            return self.pack_classifications["mixed_performance"]

    def _calculate_std(self, values: List[float], mean: float) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _assess_capacity_fade(self, avg_soh: float) -> str:
        """Assess capacity fade level"""
        fade = 100 - avg_soh
        if fade < 1:
            return "Negligible Capacity Fade (<1%)"
        elif fade < 2:
            return "Minimal Capacity Fade (1-2%)"
        elif fade < 5:
            return "Normal Capacity Fade (2-5%)"
        elif fade < 10:
            return "Moderate Capacity Fade (5-10%)"
        elif fade < 15:
            return "Significant Capacity Fade (10-15%)"
        else:
            return "Severe Capacity Fade (>15%)"

    def _assess_balancing_needs(self, std_dev: float, soh_range: float) -> str:
        """Assess cell balancing requirements"""
        if std_dev < 0.5 and soh_range < 1.0:
            return "No Balancing Required"
        elif std_dev < 1.0 and soh_range < 2.0:
            return "Preventive Balancing Recommended"
        elif std_dev < 2.0 and soh_range < 5.0:
            return "Active Balancing Required"
        elif std_dev < 3.0 and soh_range < 8.0:
            return "Intensive Balancing Protocol"
        else:
            return "Critical Balancing Intervention"

    def _determine_maintenance_priority(self, status_counts: Dict) -> str:
        """Determine maintenance priority level"""
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0 or status_counts.get(HealthStatus.END_OF_LIFE, 0) > 0:
            return "Immediate Maintenance Required"
        elif status_counts.get(HealthStatus.COMPROMISED, 0) > 0:
            return "High Priority Maintenance"
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return "Scheduled Maintenance Recommended"
        else:
            return "Routine Maintenance Sufficient"

    def _assess_operational_constraints(self, worst_soh: float) -> str:
        """Assess operational constraints"""
        if worst_soh >= 95:
            return "No Operational Constraints"
        elif worst_soh >= 90:
            return "Minor Performance Limitations"
        elif worst_soh >= 85:
            return "Moderate Operational Constraints"
        elif worst_soh >= 80:
            return "Significant Performance Constraints"
        else:
            return "Severe Operational Limitations"

    def _recommend_replacement_strategy(self, status_counts: Dict, soh_range: float) -> str:
        """Recommend replacement strategy"""
        critical_count = status_counts.get(HealthStatus.CRITICAL, 0) + status_counts.get(HealthStatus.END_OF_LIFE, 0)
        degraded_count = status_counts.get(HealthStatus.COMPROMISED, 0) + status_counts.get(HealthStatus.DEGRADED, 0)

        if critical_count > 0:
            return "Immediate Cell Replacement Required"
        elif degraded_count > len(status_counts) * 0.3:  # >30% degraded
            return "Planned Cell Replacement Program"
        elif soh_range > 10:
            return "Selective Cell Replacement"
        else:
            return "Monitor and Maintain Current Configuration"

    def _assess_thermal_risk(self, status_counts: Dict) -> str:
        """Assess thermal runaway risk"""
        critical_cells = status_counts.get(HealthStatus.CRITICAL, 0) + status_counts.get(HealthStatus.END_OF_LIFE, 0)

        if critical_cells > 0:
            return "Elevated Thermal Risk"
        elif status_counts.get(HealthStatus.COMPROMISED, 0) > 0:
            return "Moderate Thermal Risk"
        else:
            return "Low Thermal Risk"

    def _estimate_degradation_rate(self, avg_soh: float) -> str:
        """Estimate degradation rate category"""
        if avg_soh >= 98:
            return "Low Degradation Rate (<1%/year)"
        elif avg_soh >= 95:
            return "Normal Degradation Rate (1-2%/year)"
        elif avg_soh >= 90:
            return "Accelerated Degradation (2-3%/year)"
        else:
            return "High Degradation Rate (>3%/year)"

    def _assess_reliability_impact(self, worst_soh: float, std_dev: float) -> str:
        """Assess system reliability impact"""
        if worst_soh >= 95 and std_dev < 1.0:
            return "Minimal Reliability Impact"
        elif worst_soh >= 90 and std_dev < 2.0:
            return "Low Reliability Impact"
        elif worst_soh >= 85 and std_dev < 3.0:
            return "Moderate Reliability Impact"
        else:
            return "High Reliability Impact"

    def _classify_safety_status(self, worst_soh: float, soh_range: float) -> str:
        """Classify safety status"""
        if worst_soh >= 90 and soh_range < 5.0:
            return "Safe Operating Condition"
        elif worst_soh >= 85 and soh_range < 10.0:
            return "Caution Required"
        elif worst_soh >= 80:
            return "Enhanced Safety Protocols"
        else:
            return "Critical Safety Concern"

    def get_maintenance_recommendations(self, pack_analysis: Dict) -> List[str]:
        """Generate specific maintenance recommendations"""
        recommendations = []

        professional_assessment = pack_analysis.get("professional_assessment", {})
        risk_analysis = pack_analysis.get("risk_analysis", {})
        cell_distribution = pack_analysis.get("cell_distribution", {})

        # Balancing recommendations
        balancing = professional_assessment.get("balancing_requirement", "")
        if "Critical" in balancing:
            recommendations.append("Implement emergency cell balancing protocol")
        elif "Intensive" in balancing:
            recommendations.append("Schedule intensive balancing maintenance")
        elif "Active" in balancing:
            recommendations.append("Activate automated balancing system")

        # Safety recommendations
        safety = risk_analysis.get("safety_classification", "")
        if "Critical" in safety:
            recommendations.append("Implement enhanced safety monitoring")
            recommendations.append("Consider temporary operational limitations")

        # Performance recommendations
        constraints = professional_assessment.get("operational_constraints", "")
        if "Severe" in constraints:
            recommendations.append("Reduce operational load immediately")
        elif "Significant" in constraints:
            recommendations.append("Implement performance limitations")

        # Replacement recommendations
        replacement = professional_assessment.get("replacement_strategy", "")
        if "Immediate" in replacement:
            recommendations.append("Schedule emergency cell replacement")
        elif "Planned" in replacement:
            recommendations.append("Develop phased replacement program")

        return recommendations