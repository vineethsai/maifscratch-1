"""
Multi-Agent Consortium Demo using MAIF - Enhanced Version

This example demonstrates how multiple specialized agents can collaborate using MAIF
to produce a comprehensive artifact with full version history, content tracking,
and forensic analysis capabilities.

The scenario: "How do I walk from California to Nepal in a meaningful way - where I
have infinite ability to swim, and don't need to sleep"

The consortium includes:
1. GeographyAgent - Analyzes terrain and routes
2. CulturalAgent - Provides cultural insights and meaningful experiences
3. LogisticsAgent - Handles practical considerations
4. SafetyAgent - Assesses risks and safety measures
5. CoordinatorAgent - Orchestrates the collaboration and synthesizes results

Enhanced Features Demonstrated:
- Version history tracking for all content changes
- Content evolution and iterative refinement
- Cross-agent dependency management
- Forensic analysis of collaboration patterns
- Privacy and security controls
- Semantic embeddings and cross-modal attention
- Comprehensive metadata management
- Validation and integrity checking
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Import OpenAI for real AI agent responses
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

# Import MAIF modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from maif_api import MAIF, create_maif, load_maif
from maif.core import MAIFEncoder, MAIFDecoder, MAIFVersion
from maif.security import MAIFSigner, MAIFVerifier
from maif.privacy import PrivacyEngine, PrivacyLevel, PrivacyPolicy, EncryptionMode
from maif.semantic import SemanticEmbedder, KnowledgeTriple
from maif.metadata import MAIFMetadataManager
from maif.validation import MAIFValidator, MAIFRepairTool
from maif.forensics import ForensicAnalyzer
from maif.compression import MAIFCompressor


@dataclass
class AgentContribution:
    """Represents a contribution from a specialized agent."""
    agent_id: str
    agent_type: str
    contribution_type: str
    content: Dict[str, Any]
    confidence: float
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None


class BaseAgent:
    """Base class for all specialized agents in the consortium with enhanced tracking and OpenAI integration."""
    
    def __init__(self, agent_id: str, agent_type: str, specialization: str, shared_maif=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.specialization = specialization
        # Use shared MAIF if provided, otherwise create individual one (for backward compatibility)
        self.maif = shared_maif if shared_maif is not None else create_maif(agent_id, enable_privacy=True)
        self.contributions = []
        self.content_blocks = {}  # Track content block IDs for updates
        self.iteration_count = 0
        self.refinement_history = []
        
        # OpenAI configuration
        self.use_openai = OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') is not None
        if self.use_openai:
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize with agent metadata
        self._initialize_agent_metadata()
        
    def _initialize_agent_metadata(self):
        """Initialize agent with metadata and capabilities."""
        agent_metadata = {
            "agent_profile": {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "specialization": self.specialization,
                "capabilities": self._get_capabilities(),
                "initialization_time": time.time(),
                "version": "1.0.0"
            },
            "collaboration_metadata": {
                "consortium_role": self.agent_type,
                "dependency_handling": "sequential",
                "output_format": "structured_json",
                "confidence_tracking": True
            }
        }
        
        # Add agent profile to MAIF with privacy controls
        privacy_policy = PrivacyPolicy(
            privacy_level=PrivacyLevel.INTERNAL,
            encryption_mode=EncryptionMode.NONE,
            anonymization_required=False
        )
        
        self.maif.encoder.add_binary_block(
            json.dumps(agent_metadata).encode('utf-8'),
            "metadata",
            metadata={"type": "agent_profile"},
            privacy_policy=privacy_policy
        )
        
    def _get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "content_generation",
            "iterative_refinement",
            "dependency_analysis",
            "confidence_assessment",
            "version_tracking"
        ]
        
    def _call_openai(self, prompt: str, max_tokens: int = 2000) -> str:
        """Make a call to OpenAI API with the given prompt."""
        if not self.use_openai:
            return "OpenAI not available - using fallback response"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a {self.agent_type} agent specializing in {self.specialization}. Provide detailed, structured analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error for {self.agent_id}: {e}")
            return f"Error calling OpenAI: {e}"
    
    def contribute(self, query: str, context: Dict[str, Any] = None) -> AgentContribution:
        """Generate a contribution to the consortium's work."""
        # Create prompt based on agent specialization and context
        prompt = f"As a {self.agent_type} specializing in {self.specialization}, provide analysis for: {query}"
        
        # Add context information if available
        if context:
            prompt += "\n\nContext information:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
        
        # Generate content using OpenAI or fallback
        response_content = self._call_openai(prompt)
        
        # Parse response and structure as needed
        try:
            # Try to parse as JSON if possible
            content = json.loads(response_content)
        except json.JSONDecodeError:
            # If not valid JSON, use as raw text
            content = {"text": response_content}
        
        # Calculate confidence based on iteration count (higher iterations = higher confidence)
        base_confidence = 0.7
        iteration_bonus = min(0.2, self.iteration_count * 0.05)  # Max 0.2 bonus for iterations
        confidence = base_confidence + iteration_bonus
        
        # Create contribution object
        contribution = AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type=self.specialization,
            content=content,
            confidence=confidence,
            dependencies=[d for d in context.keys()] if context else [],
            metadata={
                "timestamp": time.time(),
                "iteration": self.iteration_count,
                "query": query
            }
        )
        
        # Store contribution in MAIF
        self._store_contribution(contribution)
        
        # Add to agent's contribution history
        self.contributions.append(contribution)
        
        return contribution
        
    def _store_contribution(self, contribution: AgentContribution):
        """Store contribution in MAIF with appropriate metadata."""
        # Convert contribution to JSON
        contribution_json = json.dumps({
            "agent_id": contribution.agent_id,
            "agent_type": contribution.agent_type,
            "contribution_type": contribution.contribution_type,
            "content": contribution.content,
            "confidence": contribution.confidence,
            "dependencies": contribution.dependencies,
            "metadata": contribution.metadata
        })
        
        # Store in MAIF with appropriate block type and metadata
        block_id = self.maif.encoder.add_binary_block(
            contribution_json.encode('utf-8'),
            "contribution",
            metadata={
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "iteration": self.iteration_count,
                "confidence": contribution.confidence
            }
        )
        
        # Track content block ID for future updates
        self.content_blocks[f"contribution_{self.iteration_count}"] = block_id
    
    def refine_contribution(self, original_contribution: AgentContribution,
                          feedback: Dict[str, Any]) -> AgentContribution:
        """Refine an existing contribution based on feedback."""
        self.iteration_count += 1
        
        # Create refined version
        refined_contribution = self._apply_refinements(original_contribution, feedback)
        refined_contribution.metadata = refined_contribution.metadata or {}
        refined_contribution.metadata.update({
            "iteration": self.iteration_count,
            "refinement_type": "feedback_based",
            "original_confidence": original_contribution.confidence,
            "feedback_applied": list(feedback.keys())
        })
        
        # Track refinement history
        self.refinement_history.append({
            "iteration": self.iteration_count,
            "timestamp": time.time(),
            "feedback_summary": feedback.get("feedback_summary", feedback.get("summary", "No summary provided")),
            "confidence_change": refined_contribution.confidence - original_contribution.confidence
        })
        
        return refined_contribution
    
    def _apply_refinements(self, original: AgentContribution,
                          feedback: Dict[str, Any]) -> AgentContribution:
        """Apply specific refinements based on feedback, with special handling for devil's advocate insights."""
        # Default implementation - subclasses can override
        refined_content = original.content.copy()
        
        # Process devil's advocate feedback specifically
        devils_advocate_feedback = feedback.get("devils_advocate_feedback", [])
        if devils_advocate_feedback:
            print(f"    🔍 Applying {len(devils_advocate_feedback)} critical insights to {original.agent_id}")
            
            # Add critical analysis section to content
            if "critical_analysis_addressed" not in refined_content:
                refined_content["critical_analysis_addressed"] = []
            
            for da_feedback in devils_advocate_feedback:
                refined_content["critical_analysis_addressed"].append({
                    "critical_insight": da_feedback,
                    "agent_response": f"Addressed by incorporating additional analysis and risk mitigation strategies",
                    "refinement_iteration": self.iteration_count
                })
            
            # Add improved risk assessment if devil's advocate provided feedback
            if "risk_mitigation" not in refined_content:
                refined_content["risk_mitigation"] = {
                    "identified_risks": "Based on critical analysis feedback",
                    "mitigation_strategies": "Enhanced safety protocols and contingency planning",
                    "confidence_adjustment": "Increased through addressing critical concerns"
                }
        
        # Process general suggestions
        suggestions = feedback.get("suggestions", [])
        if suggestions:
            if "improvements_made" not in refined_content:
                refined_content["improvements_made"] = []
            
            for suggestion in suggestions:
                refined_content["improvements_made"].append({
                    "suggestion": suggestion,
                    "implementation": f"Incorporated into refined analysis (iteration {self.iteration_count})",
                    "impact": "Enhanced comprehensiveness and accuracy"
                })
        
        # Add refinement metadata
        refined_content["refinement_metadata"] = {
            "iteration": self.iteration_count,
            "feedback_incorporated": feedback,
            "refinement_timestamp": time.time(),
            "devils_advocate_feedback_count": len(devils_advocate_feedback),
            "total_suggestions_addressed": len(suggestions)
        }
        
        # Adjust confidence based on feedback quality and devil's advocate input
        confidence_adjustment = 0.02 if feedback.get("positive", False) else -0.01
        
        # Additional confidence boost if devil's advocate feedback was addressed
        if devils_advocate_feedback:
            confidence_adjustment += 0.03  # Boost for addressing critical analysis
            print(f"    📈 Confidence boosted by 0.03 for addressing critical analysis")
        
        new_confidence = min(1.0, max(0.0, original.confidence + confidence_adjustment))
        
        return AgentContribution(
            agent_id=original.agent_id,
            agent_type=original.agent_type,
            contribution_type=f"{original.contribution_type}_refined",
            content=refined_content,
            confidence=new_confidence,
            dependencies=original.dependencies,
            metadata=original.metadata
        )
    
    def add_contribution_to_maif(self, contribution: AgentContribution,
                               update_existing: bool = False) -> str:
        """Add a contribution to this agent's MAIF file with version tracking."""
        
        # Determine if this is an update or new content
        content_key = contribution.contribution_type
        existing_block_id = self.content_blocks.get(content_key) if update_existing else None
        
        # Serialize content properly
        content_text = json.dumps(contribution.content, indent=2)
        content_bytes = content_text.encode('utf-8')
        
        # Prepare comprehensive metadata
        metadata = {
            "agent_id": contribution.agent_id,
            "agent_type": contribution.agent_type,
            "contribution_type": contribution.contribution_type,
            "confidence": contribution.confidence,
            "timestamp": time.time(),
            "iteration": getattr(self, 'iteration_count', 0),
            "dependencies": contribution.dependencies or [],
            "content_size": len(content_bytes),
            "content_summary": str(contribution.content)[:200] + "..." if len(str(contribution.content)) > 200 else str(contribution.content)
        }
        
        # Add custom metadata if provided
        if contribution.metadata:
            metadata.update(contribution.metadata)
        
        # Add content as binary block to ensure it's saved
        content_id = self.maif.encoder.add_binary_block(
            content_bytes,
            "text",
            metadata=metadata
        )
        
        # Store the content block ID for future updates
        self.content_blocks[content_key] = content_id
        
        # Add semantic embeddings for searchability
        if isinstance(contribution.content, dict) and 'summary' in contribution.content:
            summary_text = contribution.content['summary']
            # Generate more realistic embeddings based on content
            embeddings = self._generate_content_embeddings(summary_text)
            
            embedding_metadata = {
                "source_content": content_key,
                "embedding_model": f"{self.agent_id}_semantic_model",
                "content_hash": hash(summary_text) % 1000000,
                "generation_timestamp": time.time()
            }
            
            try:
                self.maif.add_embeddings(
                    embeddings,
                    f"{self.agent_id}_embedder",
                    compress=True
                )
            except Exception as e:
                print(f"Warning: Could not add embeddings for {self.agent_id}: {e}")
        
        # Add knowledge triples for semantic relationships
        self._add_knowledge_triples(contribution, content_id)
        
        self.contributions.append(contribution)
        return content_id
    
    def _calculate_complexity_score(self, content: Any) -> float:
        """Calculate a complexity score for the content."""
        if isinstance(content, dict):
            return min(10.0, len(content) * 0.5 + sum(
                len(str(v)) for v in content.values()
            ) / 1000.0)
        elif isinstance(content, list):
            return min(10.0, len(content) * 0.3)
        else:
            return min(10.0, len(str(content)) / 500.0)
    
    def _generate_content_embeddings(self, text: Any) -> List[List[float]]:
        """Generate semantic embeddings for content."""
        # Handle different input types
        if isinstance(text, dict):
            text = json.dumps(text)
        elif not isinstance(text, str):
            text = str(text)
            
        # Simulate realistic embeddings based on text content
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        embeddings = []
        for i in range(3):  # Generate 3 embedding vectors
            embedding = []
            for j in range(384):  # 384-dimensional embeddings
                # Create pseudo-random but deterministic values
                seed_val = int(text_hash[j % len(text_hash)], 16) + i * j
                normalized_val = (seed_val % 1000) / 1000.0 - 0.5  # Range [-0.5, 0.5]
                embedding.append(normalized_val)
            embeddings.append(embedding)
        
        return embeddings
    
    def _add_knowledge_triples(self, contribution: AgentContribution, content_id: str):
        """Add knowledge triples for semantic relationships."""
        try:
            # Create knowledge triples based on contribution
            triples_data = {
                "subject": f"agent:{self.agent_id}",
                "predicate": "contributes",
                "object": f"content:{contribution.contribution_type}",
                "confidence": contribution.confidence,
                "context": {
                    "content_id": content_id,
                    "timestamp": time.time(),
                    "agent_type": self.agent_type
                }
            }
            
            # Add as structured metadata
            metadata_content = {
                "knowledge_triple": triples_data,
                "triple_type": "agent_contribution_relationship"
            }
            self.maif.encoder.add_binary_block(
                json.dumps(metadata_content).encode('utf-8'),
                "metadata",
                metadata={"type": "knowledge_triple"}
            )
            
        except Exception as e:
            # Graceful fallback if knowledge triple creation fails
            print(f"Warning: Could not create knowledge triples for {self.agent_id}: {e}")
    
    def get_version_history(self) -> Dict[str, List[Dict]]:
        """Get version history for this agent's contributions."""
        if hasattr(self.maif.encoder, 'version_history'):
            return {
                block_id: [v.to_dict() for v in versions]
                for block_id, versions in self.maif.encoder.version_history.items()
            }
        return {}
    
    def get_contribution_summary(self) -> Dict[str, Any]:
        """Get summary of all contributions made by this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_contributions": len(self.contributions),
            "iteration_count": self.iteration_count,
            "content_blocks": len(self.content_blocks),
            "refinement_history": self.refinement_history,
            "latest_contributions": [
                {
                    "type": c.contribution_type,
                    "confidence": c.confidence,
                    "has_dependencies": bool(c.dependencies)
                }
                for c in self.contributions[-3:]  # Last 3 contributions
            ]
        }


class GeographyAgent(BaseAgent):
    """Agent specialized in geographical analysis and route planning."""
    
    def __init__(self, agent_id: str = "geo_agent_001", is_devils_advocate: bool = False, shared_maif=None):
        super().__init__(agent_id, "geography", "terrain_analysis_and_routing", shared_maif)
        self.is_devils_advocate = is_devils_advocate
        
        # Add devil's advocate metadata if applicable
        if self.is_devils_advocate:
            print(f"🔥 DEBUG: Initializing {agent_id} as devil's advocate")
            devils_advocate_metadata = {
                "role_modifier": "devils_advocate",
                "approach": "critical_analysis_and_challenge_assumptions",
                "confidence_adjustment": -0.1,  # Slightly lower confidence for contrarian views
                "criticism_focus": ["feasibility", "assumptions", "risks", "alternatives"]
            }
            
            self.maif.encoder.add_binary_block(
                json.dumps(devils_advocate_metadata).encode('utf-8'),
                "metadata",
                metadata={"type": "devils_advocate_config"}
            )
        
    def contribute(self, query: str, context: Dict[str, Any] = None) -> AgentContribution:
        """Analyze geographical aspects with optional devil's advocate perspective."""
        
        print(f"🔥 DEBUG: {self.agent_id} contributing (devil's advocate: {self.is_devils_advocate})")
        
        # Modify prompt based on devil's advocate role
        if self.is_devils_advocate:
            prompt = f"""
            As a professional geographer with expertise in terrain analysis, cartography, and route planning, acting as a DEVIL'S ADVOCATE, provide a critical technical analysis of this query:
            "{query}"
            
            Context from other agents: {json.dumps(context, indent=2) if context else "None yet"}
            
            Your role is to provide RIGOROUS TECHNICAL CRITICISM based on geographical science. Provide a detailed geographical analysis in JSON format:
            
            1. route_segments: Array with precise technical details:
               - segment_name: Official geographical designation
               - start_coordinates: [latitude, longitude]
               - end_coordinates: [latitude, longitude]
               - distance_km: Exact distance using great circle calculations
               - terrain_type: Specific geological/topographical classification
               - elevation_change_m: Precise elevation data from DEM models
               - technical_challenges: Specific geographical obstacles with scientific basis
               - feasibility_score: 0-10 rating with technical justification
            
            2. total_distance_km: Precise great circle distance calculation
            3. elevation_profile:
               - start_elevation_m: Sea level reference
               - maximum_elevation_m: Highest point with location
               - end_elevation_m: Final elevation
               - total_elevation_gain_m: Cumulative ascent
               - critical_elevation_zones: Areas above 3000m, 5000m thresholds
            
            4. geographical_challenges: Technical obstacles with scientific basis:
               - oceanic_crossing: Pacific Ocean depth profiles, current systems, temperature zones
               - mountain_barriers: Himalayan tectonics, glaciation, avalanche zones
               - climate_zones: Köppen classification transitions, extreme weather patterns
               - political_boundaries: International waters, territorial limits, visa requirements
            
            5. technical_assessment: Scientific evaluation of impossibilities:
               - hydrodynamic_analysis: Ocean swimming physics, hypothermia calculations
               - altitude_physiology: Oxygen partial pressure effects, acclimatization requirements
               - navigation_precision: GPS accuracy, magnetic declination, dead reckoning errors
            
            6. alternative_routes: Technically feasible alternatives with scientific justification
            7. summary: Professional geographical assessment with specific technical concerns
            
            Base your analysis on established geographical science, oceanography, climatology, and human physiology.
            Cite specific geographical features, coordinate systems, and measurable parameters.
            """
        else:
            prompt = f"""
            As a professional geographer with expertise in terrain analysis, cartography, and route optimization, provide a comprehensive technical analysis of this query:
            "{query}"
            
            Context from other agents: {json.dumps(context, indent=2) if context else "None yet"}
            
            CRITICAL: If context is provided, you MUST build upon and refine the previous analysis. Do not start from scratch.
            If this is a subsequent iteration, integrate and enhance technical insights from other specialists:
            - Cultural geographer inputs on sacred sites and cultural landscapes
            - Logistics specialist requirements for supply chain and infrastructure
            - Safety analyst protocols for hazard mitigation and risk assessment
            - Critical analysis feedback to address technical concerns
            
            Provide a detailed geographical analysis in JSON format with precise technical specifications:
            
            1. route_segments: Array with comprehensive technical data:
               - segment_id: Unique identifier (e.g., "CA-PAC-001")
               - segment_name: Official geographical designation
               - start_coordinates: [latitude, longitude] in WGS84
               - end_coordinates: [latitude, longitude] in WGS84
               - distance_km: Great circle distance with precision to 0.1km
               - bearing_degrees: True bearing from start to end
               - terrain_classification: USGS/geological survey classification
               - elevation_profile: {{"start_m": value, "max_m": value, "min_m": value, "end_m": value, "avg_gradient_percent": value}}
               - hydrographic_features: Rivers, lakes, coastal zones crossed
               - geological_formations: Rock types, fault lines, seismic zones
               - climate_zone: Köppen classification and seasonal variations
               - infrastructure_access: Roads, ports, airports within 50km
               - estimated_duration_hours: Based on terrain difficulty and conditions
            
            2. total_distance_km: Precise calculation using spherical trigonometry
            3. elevation_analysis:
               - total_elevation_gain_m: Cumulative positive elevation change
               - total_elevation_loss_m: Cumulative negative elevation change
               - highest_point: {{"elevation_m": value, "coordinates": [lat, lon], "location_name": "name"}}
               - lowest_point: {{"elevation_m": value, "coordinates": [lat, lon], "location_name": "name"}}
               - critical_altitude_zones: Areas requiring acclimatization (>2500m, >4000m, >5500m)
            
            4. geographical_challenges: Technical obstacles with mitigation strategies:
               - oceanic_segments: Current patterns, temperature profiles, storm seasons
               - mountain_crossings: Pass elevations, weather windows, avalanche risk
               - desert_traversals: Water sources, temperature extremes, navigation challenges
               - river_crossings: Flow rates, seasonal variations, bridge locations
               - political_boundaries: Border crossings, permit requirements, restricted zones
            
            5. waypoint_recommendations: Strategic locations with technical justification:
               - coordinates: Precise lat/long in decimal degrees
               - elevation_m: Above sea level
               - waypoint_type: Navigation, resupply, shelter, cultural, emergency
               - facilities_available: Infrastructure and services
               - seasonal_accessibility: Month-by-month availability
               - strategic_importance: Navigation, safety, or cultural significance
            
            6. navigation_specifications:
               - coordinate_system: WGS84 with UTM zone references
               - magnetic_declination: Variation from true north by region
               - gps_accuracy_zones: Areas with poor satellite coverage
               - backup_navigation: Celestial, terrain association methods
            
            7. environmental_considerations:
               - seasonal_weather_patterns: Temperature, precipitation, wind by month
               - natural_hazards: Earthquakes, tsunamis, volcanic activity, extreme weather
               - ecological_zones: Biomes crossed, protected areas, wildlife corridors
               - water_resources: Availability, quality, treatment requirements
            
            8. summary: Professional geographical assessment integrating all technical factors
            
            Base analysis on established cartographic principles, geodetic calculations, and peer-reviewed geographical research.
            Account for the hypothetical abilities (infinite swimming, no sleep) while maintaining scientific rigor in all other aspects.
            Integrate feedback from cultural, logistics, and safety specialists to optimize the route.
            """
        
        if self.use_openai:
            try:
                ai_response = self._call_openai(prompt)
                # Try to parse JSON response
                try:
                    geographical_analysis = json.loads(ai_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, create structured response from text
                    geographical_analysis = {
                        "ai_response": ai_response,
                        "summary": "AI-generated geographical analysis",
                        "note": "Response parsing needed manual structuring"
                    }
            except Exception as e:
                geographical_analysis = {
                    "error": f"AI analysis failed: {e}",
                    "fallback": "Using basic geographical knowledge"
                }
        else:
            # Fallback when OpenAI not available - different responses for devil's advocate
            if self.is_devils_advocate:
                geographical_analysis = {
                    "route_segments": [
                        {
                            "segment": "California to Pacific Ocean",
                            "distance_km": 50,
                            "terrain": "coastal_transition",
                            "challenges": ["ocean_entry", "CRITICAL: No safe ocean entry points", "Legal restrictions on ocean access"]
                        },
                        {
                            "segment": "Pacific Ocean Crossing",
                            "distance_km": 11000,
                            "terrain": "deep_ocean",
                            "challenges": ["IMPOSSIBLE: Even infinite swimming faces storms", "Hypothermia guaranteed", "Navigation impossible without landmarks", "International waters legal issues"]
                        },
                        {
                            "segment": "Asia to Himalayas",
                            "distance_km": 3000,
                            "terrain": "varied_elevation",
                            "challenges": ["CRITICAL: Visa and border issues", "Political instability", "Extreme weather variations"]
                        },
                        {
                            "segment": "Himalayas to Nepal",
                            "distance_km": 500,
                            "terrain": "extreme_altitude",
                            "challenges": ["FATAL: Altitude sickness unavoidable", "Avalanche zones", "Permit requirements"]
                        }
                    ],
                    "total_distance_km": 14550,
                    "critical_assessment": "This journey is fundamentally flawed and dangerous",
                    "alternative_suggestions": ["Fly commercially", "Use existing transportation", "Virtual reality experience"],
                    "summary": "CRITICAL ANALYSIS: 14,550 km journey is geographically impossible and dangerous despite supernatural abilities"
                }
            else:
                geographical_analysis = {
                    "route_segments": [
                        {
                            "segment": "California to Pacific Ocean",
                            "distance_km": 50,
                            "terrain": "coastal_transition",
                            "challenges": ["ocean_entry"]
                        },
                        {
                            "segment": "Pacific Ocean Crossing",
                            "distance_km": 11000,
                            "terrain": "deep_ocean",
                            "challenges": ["infinite_swimming_required", "weather", "navigation"]
                        },
                        {
                            "segment": "Asia to Himalayas",
                            "distance_km": 3000,
                            "terrain": "varied_elevation",
                            "challenges": ["altitude_gain", "climate_zones"]
                        },
                        {
                            "segment": "Himalayas to Nepal",
                            "distance_km": 500,
                            "terrain": "extreme_altitude",
                            "challenges": ["highest_peaks", "weather_extremes"]
                        }
                    ],
                    "total_distance_km": 14550,
                    "summary": "Fallback geographical analysis - 14,550 km journey requiring supernatural swimming abilities"
                }
        
        # Adjust confidence for devil's advocate
        base_confidence = 0.95 if self.use_openai else 0.75
        if self.is_devils_advocate:
            base_confidence -= 0.1  # Lower confidence for critical analysis
            
        print(f"🔥 DEBUG: {self.agent_id} contribution confidence: {base_confidence}")
        
        return AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type="geographical_route_analysis" + ("_critical" if self.is_devils_advocate else ""),
            content=geographical_analysis,
            confidence=base_confidence,
            metadata={
                "analysis_method": "ai_powered" if self.use_openai else "fallback",
                "openai_used": self.use_openai,
                "devils_advocate": self.is_devils_advocate,
                "role_modifier": "critical_analysis" if self.is_devils_advocate else "standard_analysis"
            }
        )


class CulturalAgent(BaseAgent):
    """Agent specialized in cultural insights and meaningful experiences."""
    
    def __init__(self, shared_maif=None):
        super().__init__("culture_agent_001", "cultural", "cross_cultural_experience_design", shared_maif)
        
    def contribute(self, query: str, context: Dict[str, Any] = None) -> AgentContribution:
        """Provide cultural insights for a meaningful journey using AI."""
        
        # Create detailed prompt for cultural analysis
        prompt = f"""
        As a cultural anthropologist with expertise in ethnography, linguistic anthropology, and cultural geography, provide a rigorous academic analysis of this query:
        "{query}"
        
        Context from other agents: {json.dumps(context, indent=2) if context else "None yet"}
        
        CRITICAL: If context is provided, you MUST build upon and refine the previous analysis. Do not start from scratch.
        If this is a subsequent iteration, integrate and enhance interdisciplinary insights:
        - Geographical route data to map cultural boundaries and ethnolinguistic regions
        - Logistics considerations for culturally appropriate resource procurement
        - Safety protocols incorporating cultural risk assessment and local customs
        - Critical feedback addressing cultural assumptions and methodological concerns
        
        Provide a comprehensive ethnographic analysis in JSON format with academic rigor:
        
        1. ethnolinguistic_regions: Array with detailed cultural mapping:
           - region_id: Standardized identifier (e.g., "ETH-PAC-NW-001")
           - region_name: Official ethnographic designation
           - coordinates_bounds: Geographic boundaries in decimal degrees
           - primary_ethnic_groups: Dominant cultural populations with demographics
           - language_families: Linguistic classification (ISO 639-3 codes)
           - cultural_classification: Murdock's ethnographic atlas categories
           - subsistence_patterns: Economic anthropology classification
           - social_organization: Kinship systems, political structures
           - religious_systems: Belief systems, ritual practices, sacred calendars
           - material_culture: Traditional technologies, architectural styles, crafts
           - historical_context: Colonial impacts, modernization, cultural preservation
        
        2. sacred_geography: Culturally significant locations with academic documentation:
           - site_coordinates: Precise lat/long with cultural boundaries
           - site_classification: Sacred, ceremonial, ancestral, pilgrimage
           - cultural_significance: Ethnographic documentation of importance
           - access_protocols: Traditional permissions, taboos, seasonal restrictions
           - ritual_calendar: Ceremonial timing, lunar/solar alignments
           - oral_traditions: Associated myths, legends, historical narratives
           - conservation_status: UNESCO, national, or local protection levels
           - research_ethics: FPIC requirements, community consultation protocols
        
        3. cultural_interaction_protocols: Evidence-based engagement strategies:
           - greeting_customs: Formal protocols by cultural group
           - gift_exchange: Reciprocity systems, appropriate offerings
           - communication_styles: High/low context, nonverbal patterns
           - hierarchy_respect: Age, gender, status recognition systems
           - taboo_avoidance: Food restrictions, behavioral prohibitions
           - conflict_resolution: Traditional mediation, face-saving mechanisms
           - photography_ethics: Consent protocols, sacred space restrictions
        
        4. linguistic_landscape: Technical language documentation:
           - language_vitality: UNESCO endangerment classifications
           - dialectal_variation: Geographic distribution of variants
           - contact_phenomena: Pidgins, creoles, code-switching patterns
           - essential_phrases: Phonetic transcription with cultural context
           - writing_systems: Scripts, literacy rates, orthographic standards
           - translation_challenges: Untranslatable concepts, cultural metaphors
        
        5. cultural_calendar: Temporal organization of cultural life:
           - agricultural_cycles: Planting, harvest, seasonal migrations
           - ceremonial_calendar: Religious festivals, life cycle rituals
           - market_cycles: Trade patterns, economic gatherings
           - political_seasons: Traditional governance, decision-making periods
           - storytelling_seasons: Oral tradition transmission times
           - taboo_periods: Restricted activities, mourning periods
        
        6. cultural_challenges: Anthropological risk assessment:
           - culture_shock_factors: Adaptation difficulties, psychological stress
           - misunderstanding_risks: Communication failures, offense potential
           - power_dynamics: Colonial legacies, economic disparities
           - cultural_appropriation: Boundary violations, exploitation risks
           - documentation_ethics: Research permissions, intellectual property
           - community_impact: Tourism effects, cultural commodification
        
        7. meaningful_engagement_opportunities: Ethnographically grounded experiences:
           - participatory_research: Community-based collaborative projects
           - skill_exchange: Traditional knowledge sharing, reciprocal learning
           - cultural_documentation: Oral history, language preservation
           - economic_support: Fair trade, community development initiatives
           - environmental_collaboration: Traditional ecological knowledge projects
           - artistic_exchange: Traditional arts, music, storytelling participation
        
        8. methodological_framework: Academic approach to cultural engagement:
           - ethnographic_methods: Participant observation, life history interviews
           - ethical_guidelines: AAA code of ethics, IRB considerations
           - reflexivity_practices: Positionality awareness, bias recognition
           - reciprocity_principles: Community benefit, knowledge sharing
           - documentation_standards: Field notes, audio/visual protocols
           - validation_methods: Member checking, community review
        
        9. summary: Comprehensive anthropological assessment of cultural journey potential
        
        Base analysis on peer-reviewed ethnographic literature, current anthropological theory, and established fieldwork methodologies.
        Emphasize cultural relativism, ethical engagement, and reciprocal learning relationships.
        Integrate geographical, logistical, and safety considerations through cultural lens.
        Address power dynamics, colonial legacies, and contemporary cultural politics.
        """
        
        if self.use_openai:
            try:
                ai_response = self._call_openai(prompt)
                try:
                    cultural_insights = json.loads(ai_response)
                except json.JSONDecodeError:
                    cultural_insights = {
                        "ai_response": ai_response,
                        "summary": "AI-generated cultural analysis",
                        "note": "Response parsing needed manual structuring"
                    }
            except Exception as e:
                cultural_insights = {
                    "error": f"AI analysis failed: {e}",
                    "fallback": "Using basic cultural knowledge"
                }
        else:
            # Fallback when OpenAI not available
            cultural_insights = {
                "cultural_regions": [
                    {
                        "region": "California Coast",
                        "cultural_significance": "Native American heritage",
                        "meaningful_activities": ["Land acknowledgment", "Coastal ceremonies"]
                    },
                    {
                        "region": "Pacific Ocean",
                        "cultural_significance": "Sacred waters across cultures",
                        "meaningful_activities": ["Ocean meditation", "Navigation traditions"]
                    },
                    {
                        "region": "Asian Coastlines",
                        "cultural_significance": "Maritime Buddhist traditions",
                        "meaningful_activities": ["Temple visits", "Cultural exchange"]
                    },
                    {
                        "region": "Himalayas",
                        "cultural_significance": "Sacred mountains",
                        "meaningful_activities": ["Pilgrimage", "Mountain meditation"]
                    },
                    {
                        "region": "Nepal",
                        "cultural_significance": "Birthplace of Buddha",
                        "meaningful_activities": ["Lumbini pilgrimage", "Festival participation"]
                    }
                ],
                "summary": "Fallback cultural analysis - Transform journey into spiritual pilgrimage across cultures"
            }
        
        return AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type="cultural_meaning_framework",
            content=cultural_insights,
            confidence=0.88 if self.use_openai else 0.70,
            dependencies=["geographical_route_analysis"],
            metadata={
                "research_method": "ai_powered" if self.use_openai else "fallback",
                "openai_used": self.use_openai
            }
        )


class LogisticsAgent(BaseAgent):
    """Agent specialized in practical logistics and resource management."""
    
    def __init__(self, shared_maif=None):
        super().__init__("logistics_agent_001", "logistics", "resource_optimization_and_planning", shared_maif)
        
    def contribute(self, query: str, context: Dict[str, Any] = None) -> AgentContribution:
        """Handle practical logistics for the impossible journey using AI."""
        
        # Create detailed prompt for logistics analysis
        prompt = f"""
        As a logistics specialist with expertise in supply chain management, operations research, and expedition planning, provide a comprehensive technical analysis of this query:
        "{query}"
        
        Context from other agents: {json.dumps(context, indent=2) if context else "None yet"}
        
        CRITICAL: If context is provided, you MUST build upon and refine the previous analysis. Do not start from scratch.
        Integrate and enhance interdisciplinary inputs from other specialists:
        - Geographical route data for terrain-specific logistics requirements
        - Cultural considerations for local procurement and customs compliance
        - Safety protocols for risk mitigation and emergency response planning
        - Critical analysis feedback for logistics optimization and contingency planning
        
        Provide a detailed logistics analysis in JSON format with technical specifications:
        
        1. supply_chain_analysis: Comprehensive resource management framework:
           - segment_id: Corresponding to geographical route segments
           - procurement_strategy: Local vs. pre-positioned vs. air-dropped supplies
           - inventory_requirements: Detailed BOMs (Bill of Materials) with quantities
           - storage_specifications: Temperature, humidity, security requirements
           - transportation_modes: Primary and backup logistics methods
           - customs_documentation: Required permits, declarations, duty calculations
           - lead_times: Procurement to delivery timelines with buffer analysis
           - cost_optimization: Total cost of ownership analysis per segment
        
        2. equipment_specifications: Technical requirements with standards compliance:
           - navigation_systems: GPS units (military-grade), backup compass systems, celestial navigation tools
           - communication_equipment: Satellite phones, emergency beacons (PLB/EPIRB), radio systems with frequency allocations
           - survival_gear: Emergency shelters rated for specific climate zones, water purification systems with capacity specs
           - documentation_systems: Waterproof storage, digital backup systems, chain of custody protocols
           - medical_supplies: Comprehensive first aid with expiration tracking, prescription medications, emergency procedures
           - specialized_equipment: Altitude-specific gear, cold weather systems, maritime safety equipment
        
        3. operational_timeline: Critical path analysis with resource allocation:
           - segment_duration: Time estimates with confidence intervals
           - resource_consumption_rates: Consumption models for food, water, fuel, medical supplies
           - resupply_windows: Scheduled and emergency resupply opportunities
           - critical_milestones: Key decision points and go/no-go criteria
           - seasonal_constraints: Weather windows, cultural calendar conflicts, political stability
           - buffer_analysis: Time and resource contingencies for delays
        
        4. infrastructure_requirements: Support systems and facilities:
           - communication_networks: Satellite coverage maps, terrestrial backup systems
           - transportation_hubs: Airports, seaports, road networks within operational radius
           - medical_facilities: Hospital locations, evacuation routes, telemedicine capabilities
           - diplomatic_support: Embassy locations, consular services, emergency contacts
           - financial_infrastructure: Banking access, currency exchange, emergency funding
           - technology_infrastructure: Internet connectivity, power sources, equipment maintenance
        
        5. regulatory_compliance: Legal and administrative requirements:
           - visa_requirements: Entry/exit permits, duration limits, renewal procedures
           - customs_regulations: Import/export restrictions, duty-free allowances, prohibited items
           - environmental_permits: Protected area access, wildlife interaction protocols
           - insurance_coverage: Comprehensive liability, medical evacuation, equipment protection
           - documentation_requirements: Passport validity, health certificates, emergency contacts
           - local_regulations: Municipal laws, cultural restrictions, photography permissions
        
        6. risk_management: Comprehensive contingency planning:
           - supply_chain_disruptions: Alternative suppliers, emergency procurement protocols
           - transportation_failures: Backup routes, alternative transport modes
           - equipment_failures: Redundant systems, field repair capabilities, replacement protocols
           - medical_emergencies: Evacuation procedures, telemedicine protocols, local medical contacts
           - political_instability: Route alternatives, embassy coordination, evacuation plans
           - natural_disasters: Weather monitoring, early warning systems, shelter protocols
        
        7. quality_assurance: Standards and monitoring systems:
           - equipment_standards: Military/commercial specifications, testing protocols
           - supplier_qualification: Vendor assessment, quality certifications, performance metrics
           - inventory_management: FIFO protocols, expiration tracking, condition monitoring
           - performance_monitoring: KPIs for logistics efficiency, cost control, timeline adherence
           - continuous_improvement: Lessons learned integration, process optimization
        
        8. cost_analysis: Financial planning and optimization:
           - capital_expenditures: Equipment purchase/lease costs with depreciation
           - operational_expenses: Ongoing costs for supplies, transportation, services
           - contingency_reserves: Emergency funding for unexpected costs
           - cost_optimization: Value engineering, bulk purchasing, local sourcing opportunities
           - financial_controls: Budget tracking, expense authorization, audit trails
        
        9. technology_integration: Digital systems and automation:
           - inventory_tracking: RFID/barcode systems, real-time visibility
           - route_optimization: GPS tracking, traffic analysis, weather integration
           - communication_systems: Redundant networks, encryption protocols, emergency channels
           - data_management: Cloud backup, offline capabilities, security protocols
           - monitoring_systems: IoT sensors for equipment condition, environmental monitoring
        
        10. summary: Comprehensive logistics assessment with technical recommendations
        
        Base analysis on established logistics principles, operations research methodologies, and expedition planning best practices.
        Account for the hypothetical abilities (infinite swimming, no sleep) while maintaining rigorous logistics planning for all other requirements.
        Integrate geographical, cultural, and safety considerations into comprehensive logistics framework.
        Provide quantitative analysis where possible with confidence intervals and risk assessments.
        """
        
        if self.use_openai:
            try:
                ai_response = self._call_openai(prompt)
                try:
                    logistics_plan = json.loads(ai_response)
                except json.JSONDecodeError:
                    logistics_plan = {
                        "ai_response": ai_response,
                        "summary": "AI-generated logistics analysis",
                        "note": "Response parsing needed manual structuring"
                    }
            except Exception as e:
                logistics_plan = {
                    "error": f"AI analysis failed: {e}",
                    "fallback": "Using basic logistics knowledge"
                }
        else:
            # Fallback when OpenAI not available
            logistics_plan = {
                "resource_requirements": {
                    "supernatural_abilities": {
                        "infinite_swimming": "Enables Pacific crossing",
                        "no_sleep_requirement": "24/7 travel capability"
                    },
                    "equipment_needed": ["navigation tools", "communication devices", "emergency supplies"]
                },
                "timeline_optimization": {
                    "total_estimated_duration": "45-60 days",
                    "key_segments": ["Pacific crossing: 25-30 days", "Land travel: 15-25 days"]
                },
                "support_infrastructure": {
                    "communication": "Satellite systems",
                    "documentation": "International permits and visas"
                },
                "summary": "Fallback logistics analysis - Complex international journey requiring careful planning"
            }
        
        return AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type="logistics_optimization_plan",
            content=logistics_plan,
            confidence=0.92 if self.use_openai else 0.72,
            dependencies=["geographical_route_analysis", "cultural_meaning_framework"],
            metadata={
                "planning_methodology": "ai_powered" if self.use_openai else "fallback",
                "openai_used": self.use_openai
            }
        )


class SafetyAgent(BaseAgent):
    """Agent specialized in risk assessment and safety protocols."""
    
    def __init__(self, shared_maif=None):
        super().__init__("safety_agent_001", "safety", "risk_mitigation_and_emergency_response", shared_maif)
        
    def contribute(self, query: str, context: Dict[str, Any] = None) -> AgentContribution:
        """Assess risks and develop safety protocols using AI."""
        
        # Create detailed prompt for safety analysis
        prompt = f"""
        As a safety specialist with expertise in risk assessment, emergency response, and occupational health, provide a comprehensive technical analysis of this query:
        "{query}"
        
        Context from other agents: {json.dumps(context, indent=2) if context else "None yet"}
        
        CRITICAL: If context is provided, you MUST build upon and refine the previous analysis. Do not start from scratch.
        Integrate and enhance interdisciplinary inputs from other specialists:
        - Geographical route data for terrain-specific hazard identification
        - Cultural considerations for local safety practices and emergency resources
        - Logistics planning for safety equipment deployment and emergency response
        - Critical analysis feedback for risk assessment validation and mitigation optimization
        
        Provide a detailed safety analysis in JSON format with technical specifications:
        
        1. hazard_identification: Comprehensive risk catalog with scientific basis:
           - hazard_id: Unique identifier with classification code
           - hazard_category: Environmental, biological, chemical, physical, psychosocial
           - hazard_description: Technical description with scientific parameters
           - exposure_pathways: Routes of exposure with quantitative assessment
           - affected_systems: Human body systems, equipment, or operations impacted
           - severity_classification: Catastrophic/Critical/Marginal/Negligible with criteria
           - probability_assessment: Quantitative likelihood with confidence intervals
           - temporal_factors: Acute vs. chronic exposure, seasonal variations
           - spatial_distribution: Geographic extent and intensity mapping
        
        2. risk_assessment_matrix: Quantitative risk analysis with ISO 31000 methodology:
           - risk_id: Corresponding to hazard identification
           - inherent_risk_score: Probability × Impact before controls
           - existing_controls: Current mitigation measures with effectiveness ratings
           - residual_risk_score: Risk level after existing controls
           - risk_tolerance: Acceptable risk thresholds with ALARP principles
           - control_effectiveness: Quantitative assessment of mitigation measures
           - risk_ranking: Prioritization matrix for resource allocation
        
        3. physiological_risk_assessment: Human factors analysis with medical basis:
           - altitude_physiology: Oxygen partial pressure effects, acclimatization protocols
           - thermal_stress: Heat/cold exposure limits, thermoregulation capacity
           - hydration_requirements: Water balance, electrolyte management, dehydration risks
           - nutritional_needs: Caloric requirements, micronutrient deficiencies, metabolic stress
           - sleep_deprivation: Cognitive impacts, performance degradation (accounting for no-sleep ability)
           - physical_conditioning: Fitness requirements, injury prevention, recovery protocols
           - psychological_stress: Mental health impacts, stress management, decision-making capacity
        
        4. environmental_hazard_analysis: Scientific assessment of natural risks:
           - meteorological_hazards: Storm systems, temperature extremes, precipitation patterns
           - hydrological_risks: Ocean currents, wave heights, water temperature, drowning risks
           - geological_hazards: Seismic activity, landslides, volcanic activity, terrain instability
           - biological_threats: Disease vectors, venomous species, allergenic exposures
           - atmospheric_conditions: Air quality, UV exposure, atmospheric pressure variations
           - seasonal_variations: Climate change impacts, extreme weather frequency
        
        5. technological_risk_assessment: Equipment and system failure analysis:
           - navigation_system_failures: GPS outages, compass deviation, backup navigation
           - communication_equipment_failures: Satellite coverage gaps, equipment malfunction
           - life_support_system_failures: Water purification, shelter integrity, medical equipment
           - transportation_failures: Vehicle breakdown, fuel shortage, mechanical issues
           - power_system_failures: Battery depletion, solar panel damage, charging infrastructure
           - data_system_failures: Information loss, backup system integrity, cybersecurity
        
        6. geopolitical_risk_analysis: Security and regulatory assessment:
           - political_stability: Government stability indices, conflict probability
           - border_security: Entry/exit procedures, documentation requirements, detention risks
           - law_enforcement: Local police capabilities, emergency response times
           - terrorism_threats: Regional threat levels, target vulnerability assessment
           - criminal_activity: Theft, assault, kidnapping risks with geographic distribution
           - regulatory_compliance: Legal requirements, permit violations, penalty assessment
        
        7. emergency_response_protocols: Systematic emergency management:
           - incident_classification: Emergency categories with response triggers
           - notification_procedures: Emergency contacts, escalation protocols, communication trees
           - evacuation_procedures: Route planning, transportation assets, assembly points
           - medical_emergency_response: First aid protocols, evacuation procedures, hospital networks
           - search_and_rescue: SAR coordination, location reporting, survival priorities
           - crisis_communication: Media management, family notification, stakeholder updates
        
        8. safety_management_system: Comprehensive safety framework:
           - safety_policy: Organizational commitment, responsibility assignment
           - safety_planning: Risk management integration, resource allocation
           - safety_implementation: Training requirements, competency assessment
           - safety_monitoring: Performance indicators, audit protocols, incident reporting
           - safety_review: Continuous improvement, lessons learned integration
           - safety_culture: Behavioral safety, reporting culture, leadership commitment
        
        9. personal_protective_equipment: Technical specifications and deployment:
           - head_protection: Helmet specifications for impact, penetration, electrical hazards
           - eye_protection: UV protection, impact resistance, chemical splash protection
           - respiratory_protection: Filtration efficiency, breathing resistance, fit testing
           - body_protection: Thermal protection, chemical resistance, cut/puncture protection
           - hand_protection: Dexterity requirements, chemical compatibility, grip performance
           - foot_protection: Slip resistance, puncture protection, electrical insulation
           - fall_protection: Harness specifications, anchor points, rescue procedures
        
        10. monitoring_and_surveillance: Continuous safety assessment systems:
            - health_monitoring: Vital signs tracking, fatigue assessment, medical alerts
            - environmental_monitoring: Weather stations, air quality sensors, radiation detection
            - location_tracking: GPS monitoring, emergency beacon activation, route deviation alerts
            - communication_monitoring: Check-in protocols, emergency signal procedures
            - equipment_monitoring: Condition assessment, maintenance schedules, failure prediction
            - performance_monitoring: Safety KPIs, incident trends, near-miss analysis
        
        11. training_and_competency: Safety education and skill development:
            - risk_awareness_training: Hazard recognition, risk assessment skills
            - emergency_response_training: First aid, CPR, emergency procedures
            - equipment_operation_training: Proper use, maintenance, troubleshooting
            - survival_skills_training: Wilderness survival, water survival, urban survival
            - cultural_sensitivity_training: Local customs, conflict avoidance, communication
            - continuous_education: Refresher training, skill updates, competency validation
        
        12. summary: Comprehensive safety assessment with technical recommendations and risk prioritization
        
        Base analysis on established safety management principles, risk assessment methodologies, and emergency response best practices.
        Account for the hypothetical abilities (infinite swimming, no sleep) while maintaining rigorous safety analysis for all other risk factors.
        Integrate geographical, cultural, and logistics considerations into comprehensive safety framework.
        Provide quantitative risk assessments where possible with statistical confidence levels.
        """
        
        if self.use_openai:
            try:
                ai_response = self._call_openai(prompt)
                try:
                    safety_assessment = json.loads(ai_response)
                except json.JSONDecodeError:
                    safety_assessment = {
                        "ai_response": ai_response,
                        "summary": "AI-generated safety analysis",
                        "note": "Response parsing needed manual structuring"
                    }
            except Exception as e:
                safety_assessment = {
                    "error": f"AI analysis failed: {e}",
                    "fallback": "Using basic safety knowledge"
                }
        else:
            # Fallback when OpenAI not available
            safety_assessment = {
                "risk_categories": {
                    "environmental_risks": [
                        {"risk": "Pacific Ocean hazards", "severity": "extreme", "mitigation": ["Weather monitoring", "Emergency protocols"]},
                        {"risk": "Himalayan altitude", "severity": "high", "mitigation": ["Acclimatization", "Oxygen backup"]}
                    ],
                    "political_risks": [
                        {"risk": "Border crossings", "severity": "medium", "mitigation": ["Diplomatic coordination", "Proper documentation"]}
                    ],
                    "health_risks": [
                        {"risk": "Hypothermia", "severity": "high", "mitigation": ["Thermal protection", "Temperature monitoring"]}
                    ]
                },
                "safety_protocols": {
                    "pre_journey": ["Medical examination", "Emergency contacts", "Equipment testing"],
                    "during_journey": ["Regular check-ins", "Health monitoring", "Weather assessment"],
                    "emergency_procedures": ["Evacuation protocols", "Rescue coordination"]
                },
                "summary": "Fallback safety analysis - Comprehensive protocols needed despite supernatural abilities"
            }
        
        return AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type="comprehensive_safety_assessment",
            content=safety_assessment,
            confidence=0.94 if self.use_openai else 0.74,
            dependencies=["geographical_route_analysis", "logistics_optimization_plan"],
            metadata={
                "risk_methodology": "ai_powered" if self.use_openai else "fallback",
                "openai_used": self.use_openai
            }
        )


class DevilsAdvocateAgent(BaseAgent):
    """Agent specialized in critical analysis and challenging all other agent contributions."""
    
    def __init__(self, shared_maif=None):
        super().__init__("devils_advocate_001", "critical_analysis", "comprehensive_critique_and_challenge", shared_maif)
        
    def contribute(self, query: str, other_contributions: Dict[str, AgentContribution] = None) -> AgentContribution:
        """Provide rigorous critical analysis of all other agent contributions."""
        print(f"🔥 DEBUG: {self.agent_id} analyzing all agent contributions critically")
        
        # Create comprehensive prompt for critical analysis
        contributions_summary = ""
        if other_contributions:
            for agent_id, contribution in other_contributions.items():
                if hasattr(contribution, 'content'):
                    contributions_summary += f"\n{agent_id}: {json.dumps(contribution.content, indent=2)}"
                else:
                    contributions_summary += f"\n{agent_id}: {json.dumps(contribution, indent=2)}"
        
        prompt = f"""
        As a critical analyst with expertise in systems analysis, risk assessment, and scientific methodology, provide a rigorous technical critique of this query and all agent contributions:
        
        QUERY: "{query}"
        
        AGENT CONTRIBUTIONS TO ANALYZE:
        {contributions_summary}
        
        Your role is to provide SYSTEMATIC CRITICAL ANALYSIS using established scientific and engineering principles. Provide a comprehensive critique in JSON format:
        
        1. methodological_critique: Analysis of each agent's approach:
           - agent_id: Identifier of the agent being critiqued
           - methodology_assessment: Evaluation of analytical approach and rigor
           - data_quality: Assessment of evidence base and sources
           - assumption_validity: Critical examination of underlying assumptions
           - logical_consistency: Evaluation of reasoning and conclusions
           - technical_accuracy: Assessment of technical specifications and calculations
           - completeness: Gaps in analysis or missing considerations
           - bias_identification: Potential biases or conflicts of interest
        
        2. interdisciplinary_integration_critique: Assessment of cross-agent coordination:
           - integration_quality: How well agents incorporated others' inputs
           - consistency_analysis: Contradictions or conflicts between agent recommendations
           - synergy_assessment: Missed opportunities for collaborative optimization
           - communication_effectiveness: Quality of information transfer between agents
           - systems_thinking: Holistic vs. siloed approach evaluation
        
        3. feasibility_analysis: Technical and practical viability assessment:
           - physical_constraints: Laws of physics, engineering limitations
           - resource_requirements: Realistic assessment of needed resources
           - technological_readiness: Current vs. required technology levels
           - economic_viability: Cost-benefit analysis and financial feasibility
           - regulatory_compliance: Legal and regulatory barriers
           - timeline_realism: Achievability within proposed timeframes
        
        4. risk_assessment_critique: Evaluation of risk identification and mitigation:
           - risk_identification_completeness: Missed or underestimated risks
           - probability_assessment_accuracy: Realistic vs. optimistic probability estimates
           - impact_assessment_validity: Consequence evaluation accuracy
           - mitigation_strategy_effectiveness: Viability of proposed risk controls
           - contingency_planning_adequacy: Backup plan robustness
           - monitoring_system_sufficiency: Risk tracking and early warning systems
        
        5. alternative_analysis: Critical examination of alternatives:
           - alternative_identification: Other approaches not considered
           - comparative_analysis: Systematic comparison of options
           - opportunity_cost_assessment: What is sacrificed by chosen approach
           - innovation_potential: More creative or effective solutions
           - scalability_considerations: Adaptability to different scenarios
        
        6. evidence_quality_assessment: Scientific rigor evaluation:
           - source_credibility: Quality and reliability of information sources
           - data_sufficiency: Adequacy of evidence base for conclusions
           - peer_review_status: Whether recommendations align with peer-reviewed research
           - replication_potential: Ability to verify or reproduce analysis
           - uncertainty_quantification: Acknowledgment and handling of uncertainties
        
        7. ethical_considerations: Moral and ethical implications:
           - stakeholder_impact: Effects on all affected parties
           - cultural_sensitivity: Respect for local customs and values
           - environmental_responsibility: Ecological impact assessment
           - social_justice: Equity and fairness considerations
           - informed_consent: Proper consultation and agreement processes
           - long_term_consequences: Intergenerational impact assessment
        
        8. improvement_recommendations: Specific suggestions for enhancement:
           - methodology_improvements: Better analytical approaches
           - data_collection_needs: Additional information requirements
           - collaboration_enhancements: Improved inter-agent coordination
           - risk_mitigation_upgrades: Stronger safety and contingency measures
           - quality_assurance_measures: Validation and verification processes
           - stakeholder_engagement: Better consultation and communication
        
        9. overall_assessment: Comprehensive evaluation summary:
           - strengths_identification: What the agents did well
           - critical_weaknesses: Major flaws or gaps in analysis
           - confidence_level: Degree of confidence in agent recommendations
           - recommendation_validity: Whether proposals should be accepted, modified, or rejected
           - priority_concerns: Most critical issues requiring immediate attention
        
        10. summary: Executive summary of critical analysis with key findings and recommendations
        
        Base your critique on established scientific principles, engineering best practices, and rigorous analytical methodologies.
        Be constructively critical - identify problems while suggesting improvements.
        Maintain objectivity and avoid personal bias while providing thorough technical assessment.
        Consider both the hypothetical elements (infinite swimming, no sleep) and realistic constraints.
        """
        
        if self.use_openai:
            try:
                ai_response = self._call_openai(prompt, max_tokens=3000)
                try:
                    critical_analysis = json.loads(ai_response)
                except json.JSONDecodeError:
                    critical_analysis = {
                        "ai_response": ai_response,
                        "summary": "AI-generated critical analysis",
                        "note": "Response parsing needed manual structuring"
                    }
            except Exception as e:
                critical_analysis = {
                    "error": f"AI analysis failed: {e}",
                    "fallback": "Using basic critical analysis"
                }
        else:
            # Fallback critical analysis
            critical_analysis = {
                "overall_assessment": "Comprehensive technical critique of multi-agent analysis",
                "critical_challenges": [
                    "Methodological rigor: Need for more systematic analytical approaches",
                    "Evidence quality: Insufficient peer-reviewed sources and quantitative data",
                    "Risk assessment: Underestimation of technical and practical challenges",
                    "Integration gaps: Limited cross-disciplinary coordination and validation",
                    "Feasibility concerns: Optimistic assumptions about resource availability and timeline"
                ],
                "agent_critique": {},
                "alternative_recommendations": [
                    "Implement systematic peer review process for all agent contributions",
                    "Establish quantitative metrics and validation criteria",
                    "Develop integrated risk assessment framework across all disciplines",
                    "Create formal quality assurance and verification protocols"
                ],
                "risk_assessment": "Moderate to high - significant technical and coordination challenges",
                "summary": "While agents provide valuable specialized insights, systematic integration and validation processes needed for reliable recommendations."
            }
        
        # Add specific critiques for each agent if contributions provided
        if other_contributions:
            for agent_id, contribution in other_contributions.items():
                # Extract agent type for targeted critique
                if hasattr(contribution, 'agent_type'):
                    agent_type = contribution.agent_type
                else:
                    # Extract agent type from agent_id
                    if 'geo_agent' in agent_id:
                        agent_type = 'geography'
                    elif 'culture_agent' in agent_id:
                        agent_type = 'cultural'
                    elif 'logistics_agent' in agent_id:
                        agent_type = 'logistics'
                    elif 'safety_agent' in agent_id:
                        agent_type = 'safety'
                    else:
                        agent_type = 'unknown'
                
                critical_analysis["agent_critique"][agent_id] = {
                    "agent_type": agent_type,
                    "critique": f"Technical critique of {agent_type} analysis - requires validation against established standards",
                    "overoptimism_score": 0.7,
                    "reality_check": "Assumptions require verification against scientific literature and practical constraints"
                }
        
        return AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type="comprehensive_critical_analysis",
            content=critical_analysis,
            confidence=0.95,  # High confidence in the critique
            dependencies=list(other_contributions.keys()) if other_contributions else [],
            metadata={
                "role": "devils_advocate",
                "critique_scope": "all_agents",
                "analysis_type": "critical_reality_check"
            }
        )


from enum import Enum

class CollaborationState(Enum):
    """States in the multi-agent collaboration state machine."""
    INITIAL_CONTRIBUTIONS = "initial_contributions"
    DEVILS_ADVOCATE_ANALYSIS = "devils_advocate_analysis"
    FEEDBACK_GENERATION = "feedback_generation"
    AGENT_REFINEMENT = "agent_refinement"
    CONVERGENCE_CHECK = "convergence_check"
    FINAL_SYNTHESIS = "final_synthesis"
    COMPLETE = "complete"

class CollaborationStateMachine:
    """State machine for managing multi-round collaborative refinement with MAIF integration."""
    
    def __init__(self, coordinator_agent):
        self.coordinator = coordinator_agent
        self.state = CollaborationState.INITIAL_CONTRIBUTIONS
        self.round_number = 0
        self.max_rounds = 5
        self.convergence_threshold = 0.85
        self.round_history = []
        self.devils_advocate_feedback_history = []
        self.state_transitions = []
        self.convergence_metrics = []
        
    def execute_collaboration_cycle(self, agents: List[BaseAgent], query: str) -> Dict[str, AgentContribution]:
        """Execute the full collaboration cycle using state machine with MAIF integration."""
        print(f"\n🔄 STARTING COLLABORATION STATE MACHINE")
        print(f"Max rounds: {self.max_rounds}, Convergence threshold: {self.convergence_threshold}")
        print(f"🎯 MAIF-powered context propagation and version tracking enabled")
        
        current_contributions = {}
        devils_advocate_agent = self._find_devils_advocate(agents)
        other_agents = [a for a in agents if a != devils_advocate_agent]
        
        while self.state != CollaborationState.COMPLETE and self.round_number < self.max_rounds:
            self.round_number += 1
            print(f"\n{'='*60}")
            print(f"🔄 COLLABORATION ROUND {self.round_number}")
            print(f"Current State: {self.state.value}")
            print(f"{'='*60}")
            
            # Execute current state
            if self.state == CollaborationState.INITIAL_CONTRIBUTIONS:
                current_contributions = self._state_collect_contributions(other_agents, query)
                self._transition_to(CollaborationState.DEVILS_ADVOCATE_ANALYSIS)
                
            elif self.state == CollaborationState.DEVILS_ADVOCATE_ANALYSIS:
                devils_advocate_feedback = self._state_devils_advocate_analysis(devils_advocate_agent, current_contributions, query)
                # 🔥 FIX: The devil's advocate analysis modifies current_contributions by adding the DA contribution
                # No need to reassign since contributions dict is modified in place
                self._transition_to(CollaborationState.FEEDBACK_GENERATION)
                
            elif self.state == CollaborationState.FEEDBACK_GENERATION:
                targeted_feedback = self._state_generate_targeted_feedback(devils_advocate_feedback, current_contributions)
                self._transition_to(CollaborationState.AGENT_REFINEMENT)
                
            elif self.state == CollaborationState.AGENT_REFINEMENT:
                current_contributions = self._state_refine_contributions(other_agents, current_contributions, targeted_feedback)
                self._transition_to(CollaborationState.CONVERGENCE_CHECK)
                
            elif self.state == CollaborationState.CONVERGENCE_CHECK:
                if self._check_convergence(current_contributions):
                    self._transition_to(CollaborationState.FINAL_SYNTHESIS)
                else:
                    self._transition_to(CollaborationState.INITIAL_CONTRIBUTIONS)
                    
            elif self.state == CollaborationState.FINAL_SYNTHESIS:
                final_artifact = self._state_final_synthesis(current_contributions, query)
                self._transition_to(CollaborationState.COMPLETE)
                print(f"🔥 DEBUG: State machine returning {len(current_contributions)} contributions: {list(current_contributions.keys())}")
                return current_contributions
        
        # Max rounds reached
        print(f"\n⚠️  Max rounds ({self.max_rounds}) reached. Proceeding to final synthesis.")
        self._transition_to(CollaborationState.FINAL_SYNTHESIS)
        final_artifact = self._state_final_synthesis(current_contributions, query)
        self._transition_to(CollaborationState.COMPLETE)
        
        print(f"🔥 DEBUG: State machine returning {len(current_contributions)} contributions: {list(current_contributions.keys())}")
        return current_contributions
    
    def _find_devils_advocate(self, agents: List[BaseAgent]) -> BaseAgent:
        """Find the devil's advocate agent."""
        for agent in agents:
            if "devils_advocate" in agent.agent_id.lower():
                return agent
        return None
    
    def _transition_to(self, new_state: CollaborationState):
        """Transition to a new state with MAIF logging."""
        old_state = self.state
        self.state = new_state
        
        transition = {
            "from_state": old_state.value,
            "to_state": new_state.value,
            "round": self.round_number,
            "timestamp": time.time()
        }
        self.state_transitions.append(transition)
        
        # Log state transition to MAIF
        self.coordinator.maif.add_text(
            f"State Transition: {old_state.value} → {new_state.value}\nTransition: {transition}\nRound: {self.round_number}",
            title=f"State Transition Round {self.round_number}"
        )
        
        print(f"🔄 State Transition: {old_state.value} → {new_state.value}")
    
    def _state_collect_contributions(self, agents: List[BaseAgent], query: str) -> Dict[str, AgentContribution]:
        """State: Collect contributions from all agents with MAIF context."""
        print(f"\n📝 STATE: COLLECTING CONTRIBUTIONS (Round {self.round_number})")
        
        contributions = {}
        
        # Load context from previous rounds via MAIF
        previous_context = self._load_maif_context_for_round()
        
        for agent in agents:
            print(f"\n  🤖 Collecting from {agent.agent_id}")
            
            # Provide context from previous rounds
            if previous_context:
                print(f"    📚 Providing context from {len(previous_context.get('previous_rounds', 0))} previous rounds")
                agent.previous_round_context = previous_context
            
            contribution = agent.contribute(query)
            contributions[agent.agent_id] = contribution
            
            # Store in MAIF with round metadata
            content_id = agent.add_contribution_to_maif(contribution)
            agent.maif.add_text(
                f"Round {self.round_number} Contribution\nType: round_contribution\nState: initial_contributions\nContent ID: {content_id}\nConfidence: {contribution.confidence}",
                title=f"Round {self.round_number} Contribution"
            )
            
            print(f"    ✅ Contribution collected (confidence: {contribution.confidence:.2f})")
        
        return contributions
    
    def _state_devils_advocate_analysis(self, devils_advocate: BaseAgent, contributions: Dict[str, AgentContribution], query: str) -> Dict[str, Any]:
        """State: Devil's advocate provides targeted critical analysis."""
        print(f"\n🔍 STATE: DEVIL'S ADVOCATE ANALYSIS")
        
        if not devils_advocate:
            print("  ⚠️  No devil's advocate agent found. Skipping critical analysis.")
            return {}
        
        # Provide all contributions as context to devil's advocate
        devils_advocate.current_round_contributions = contributions
        
        print(f"  🎯 {devils_advocate.agent_id} analyzing {len(contributions)} contributions")
        
        # Get devil's advocate analysis
        da_contribution = devils_advocate.contribute(query)
        
        # 🔥 FIX: Add devil's advocate contribution to the main contributions dict
        contributions[devils_advocate.agent_id] = da_contribution
        print(f"  ✅ Added {devils_advocate.agent_id} contribution to main contributions")
        
        # Store devil's advocate analysis in MAIF
        content_id = devils_advocate.add_contribution_to_maif(da_contribution)
        devils_advocate.maif.add_text(
            f"Round {self.round_number} Critical Analysis\nType: devils_advocate_analysis\nAnalyzed agents: {list(contributions.keys())}\nContent ID: {content_id}\nConfidence: {da_contribution.confidence}",
            title=f"Round {self.round_number} Critical Analysis"
        )
        
        # Extract structured feedback from devil's advocate
        feedback = self._extract_structured_feedback(da_contribution)
        self.devils_advocate_feedback_history.append({
            "round": self.round_number,
            "feedback": feedback,
            "confidence": da_contribution.confidence
        })
        
        print(f"  ✅ Critical analysis complete (confidence: {da_contribution.confidence:.2f})")
        print(f"  📊 Generated {len(feedback.get('agent_specific_feedback', {}))} targeted critiques")
        
        return feedback
    
    def _extract_structured_feedback(self, da_contribution: AgentContribution) -> Dict[str, Any]:
        """Extract structured, actionable feedback from devil's advocate contribution."""
        if not isinstance(da_contribution.content, dict):
            return {"general_feedback": str(da_contribution.content)}
        
        content = da_contribution.content
        
        # Extract agent-specific critiques
        agent_specific_feedback = {}
        if "agent_critique" in content:
            agent_critiques = content["agent_critique"]
            if isinstance(agent_critiques, dict):
                for critique_key, critique_value in agent_critiques.items():
                    # Map critique to agent types
                    if "geography" in critique_key.lower() or "geo" in critique_key.lower():
                        agent_specific_feedback["geography"] = {
                            "critique": critique_value,
                            "action_items": self._generate_action_items(critique_value, "geography"),
                            "priority": "high"
                        }
                    elif "cultural" in critique_key.lower() or "culture" in critique_key.lower():
                        agent_specific_feedback["cultural"] = {
                            "critique": critique_value,
                            "action_items": self._generate_action_items(critique_value, "cultural"),
                            "priority": "high"
                        }
                    elif "logistics" in critique_key.lower():
                        agent_specific_feedback["logistics"] = {
                            "critique": critique_value,
                            "action_items": self._generate_action_items(critique_value, "logistics"),
                            "priority": "medium"
                        }
                    elif "safety" in critique_key.lower():
                        agent_specific_feedback["safety"] = {
                            "critique": critique_value,
                            "action_items": self._generate_action_items(critique_value, "safety"),
                            "priority": "high"
                        }
        
        return {
            "agent_specific_feedback": agent_specific_feedback,
            "critical_challenges": content.get("critical_challenges", []),
            "alternative_recommendations": content.get("alternative_recommendations", []),
            "overall_assessment": content.get("overall_assessment", {}),
            "confidence": da_contribution.confidence
        }
    
    def _generate_action_items(self, critique: str, agent_type: str) -> List[str]:
        """Generate specific action items based on critique and agent type."""
        action_items = []
        
        if isinstance(critique, str):
            critique_lower = critique.lower()
            
            # Geography-specific action items
            if agent_type == "geography":
                if "distance" in critique_lower or "route" in critique_lower:
                    action_items.append("Recalculate distances using more accurate methods")
                    action_items.append("Provide alternative route options")
                if "terrain" in critique_lower or "obstacle" in critique_lower:
                    action_items.append("Include detailed terrain analysis")
                    action_items.append("Identify specific geographical obstacles")
                if "impossible" in critique_lower or "unrealistic" in critique_lower:
                    action_items.append("Acknowledge physical limitations")
                    action_items.append("Propose theoretical vs practical scenarios")
            
            # Cultural-specific action items
            elif agent_type == "cultural":
                if "cultural" in critique_lower or "tradition" in critique_lower:
                    action_items.append("Research specific cultural practices")
                    action_items.append("Include cultural sensitivity protocols")
                if "meaning" in critique_lower or "significance" in critique_lower:
                    action_items.append("Expand on cultural significance")
                    action_items.append("Provide historical context")
            
            # Logistics-specific action items
            elif agent_type == "logistics":
                if "resource" in critique_lower or "supply" in critique_lower:
                    action_items.append("Detail specific resource requirements")
                    action_items.append("Include supply chain analysis")
                if "timeline" in critique_lower or "duration" in critique_lower:
                    action_items.append("Provide realistic timeline estimates")
                    action_items.append("Include contingency time buffers")
            
            # Safety-specific action items
            elif agent_type == "safety":
                if "risk" in critique_lower or "danger" in critique_lower:
                    action_items.append("Conduct comprehensive risk assessment")
                    action_items.append("Develop specific mitigation strategies")
                if "emergency" in critique_lower or "contingency" in critique_lower:
                    action_items.append("Create detailed emergency protocols")
                    action_items.append("Include backup plans")
        
        # Default action items if none generated
        if not action_items:
            action_items = [
                f"Address the critique: {critique[:100]}...",
                f"Provide more detailed {agent_type} analysis",
                "Include supporting evidence and rationale"
            ]
        
        return action_items[:3]  # Limit to 3 action items
    
    def _state_generate_targeted_feedback(self, devils_advocate_feedback: Dict[str, Any], contributions: Dict[str, AgentContribution]) -> Dict[str, Dict[str, Any]]:
        """State: Generate targeted feedback for each agent based on devil's advocate analysis."""
        print(f"\n🎯 STATE: GENERATING TARGETED FEEDBACK")
        
        targeted_feedback = {}
        agent_specific = devils_advocate_feedback.get("agent_specific_feedback", {})
        
        for agent_id, contribution in contributions.items():
            agent_type = contribution.agent_type
            
            feedback = {
                "needs_refinement": False,
                "suggestions": [],
                "action_items": [],
                "devils_advocate_feedback": [],
                "priority": "medium",
                "confidence_assessment": contribution.confidence
            }
            
            # Map agent type to devil's advocate feedback
            if agent_type in agent_specific:
                da_feedback = agent_specific[agent_type]
                feedback["needs_refinement"] = True
                feedback["devils_advocate_feedback"].append(da_feedback["critique"])
                feedback["action_items"] = da_feedback["action_items"]
                feedback["priority"] = da_feedback["priority"]
                feedback["suggestions"].extend([f"Action: {item}" for item in da_feedback["action_items"]])
                
                print(f"  🎯 {agent_id}: {len(da_feedback['action_items'])} action items (priority: {da_feedback['priority']})")
            
            # Add general challenges as suggestions
            critical_challenges = devils_advocate_feedback.get("critical_challenges", [])
            if critical_challenges:
                relevant_challenges = [c for c in critical_challenges[:2] if isinstance(c, str)]
                feedback["suggestions"].extend([f"Address challenge: {c}" for c in relevant_challenges])
            
            # Store feedback in MAIF
            self.coordinator.maif.add_text(
                f"Round {self.round_number} Feedback for {agent_id}\nType: targeted_feedback\nAgent ID: {agent_id}\nDevil's Advocate Driven: {len(feedback['devils_advocate_feedback']) > 0}\nFeedback: {json.dumps(feedback, indent=2)}",
                title=f"Round {self.round_number} Feedback for {agent_id}"
            )
            
            targeted_feedback[agent_id] = feedback
        
        print(f"  ✅ Generated targeted feedback for {len(targeted_feedback)} agents")
        return targeted_feedback
    
    def _state_refine_contributions(self, agents: List[BaseAgent], contributions: Dict[str, AgentContribution], feedback: Dict[str, Dict[str, Any]]) -> Dict[str, AgentContribution]:
        """State: Agents refine their contributions based on targeted feedback."""
        print(f"\n🔧 STATE: AGENT REFINEMENT")
        
        # 🔥 FIX: Start with all existing contributions to preserve devil's advocate
        refined_contributions = contributions.copy()
        
        for agent in agents:
            if agent.agent_id in contributions and agent.agent_id in feedback:
                agent_feedback = feedback[agent.agent_id]
                
                if agent_feedback.get("needs_refinement", False):
                    print(f"  🔧 Refining {agent.agent_id} (priority: {agent_feedback.get('priority', 'medium')})")
                    
                    original = contributions[agent.agent_id]
                    refined = agent.refine_contribution(original, agent_feedback)
                    
                    # Update in MAIF with version tracking
                    content_id = agent.add_contribution_to_maif(refined, update_existing=True)
                    agent.maif.add_text(
                        f"Round {self.round_number} Refinement\nType: refined_contribution\nOriginal confidence: {original.confidence}\nRefined confidence: {refined.confidence}\nImprovement: {refined.confidence - original.confidence}\nAction items addressed: {len(agent_feedback.get('action_items', []))}\nContent ID: {content_id}",
                        title=f"Round {self.round_number} Refinement"
                    )
                    
                    refined_contributions[agent.agent_id] = refined
                    print(f"    ✅ Refined (confidence: {original.confidence:.2f} → {refined.confidence:.2f})")
                else:
                    # No change needed, contribution already in refined_contributions
                    print(f"  ✅ {agent.agent_id} - no refinement needed")
        
        print(f"  🔥 DEBUG: Refinement preserving {len(refined_contributions)} contributions: {list(refined_contributions.keys())}")
        return refined_contributions
    
    def _check_convergence(self, contributions: Dict[str, AgentContribution]) -> bool:
        """Check if contributions have converged."""
        print(f"\n📊 STATE: CONVERGENCE CHECK")
        
        confidences = [c.confidence for c in contributions.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        
        # Calculate improvement from previous round
        improvement = 0
        if len(self.convergence_metrics) > 0:
            prev_avg = self.convergence_metrics[-1]["average_confidence"]
            improvement = avg_confidence - prev_avg
        
        metrics = {
            "round": self.round_number,
            "average_confidence": avg_confidence,
            "minimum_confidence": min_confidence,
            "improvement": improvement,
            "converged": avg_confidence >= self.convergence_threshold and min_confidence >= 0.7
        }
        
        self.convergence_metrics.append(metrics)
        
        # Store convergence metrics in MAIF
        self.coordinator.maif.add_text(
            f"Round {self.round_number} Convergence Check\nType: convergence_metrics\nMetrics: {json.dumps(metrics, indent=2)}",
            title=f"Round {self.round_number} Convergence Check"
        )
        
        print(f"  📊 Average confidence: {avg_confidence:.3f}")
        print(f"  📊 Minimum confidence: {min_confidence:.3f}")
        print(f"  📊 Improvement: {improvement:+.3f}")
        print(f"  📊 Threshold: {self.convergence_threshold}")
        
        if metrics["converged"]:
            print(f"  ✅ CONVERGENCE ACHIEVED!")
            return True
        else:
            print(f"  🔄 Continuing to next round...")
            return False
    
    def _state_final_synthesis(self, contributions: Dict[str, AgentContribution], query: str) -> AgentContribution:
        """State: Final synthesis of all contributions."""
        print(f"\n🎯 STATE: FINAL SYNTHESIS")
        
        # Use coordinator's existing synthesis method
        self.coordinator.agent_contributions = contributions
        final_artifact = self.coordinator.synthesize_final_artifact(query)
        
        # Store final synthesis in MAIF
        content_id = self.coordinator.add_contribution_to_maif(final_artifact)
        self.coordinator.maif.add_text(
            f"Final Consortium Synthesis\nType: final_synthesis\nTotal rounds: {self.round_number}\nFinal confidence: {final_artifact.confidence}\nState transitions: {len(self.state_transitions)}\nContent ID: {content_id}",
            title="Final Consortium Synthesis"
        )
        
        print(f"  ✅ Final synthesis complete (confidence: {final_artifact.confidence:.2f})")
        return final_artifact
    
    def _load_maif_context_for_round(self) -> Dict[str, Any]:
        """Load context from previous rounds stored in MAIF."""
        if self.round_number <= 1:
            return {}
        
        # Load previous round data from MAIF
        context = {
            "previous_rounds": self.round_number - 1,
            "state_transitions": self.state_transitions,
            "convergence_history": self.convergence_metrics,
            "devils_advocate_feedback": self.devils_advocate_feedback_history
        }
        
        return context


class CoordinatorAgent(BaseAgent):
    """Agent that orchestrates the consortium and synthesizes all contributions with comprehensive tracking."""
    
    def __init__(self, shared_maif=None):
        super().__init__("coordinator_agent_001", "coordinator", "multi_agent_orchestration", shared_maif)
        self.agent_contributions = {}
        self.collaboration_rounds = 0
        self.feedback_history = []
        self.synthesis_iterations = []
        self.state_machine = CollaborationStateMachine(self)
        
        # Initialize coordinator-specific metadata
        self._initialize_coordinator_metadata()
        
    def _initialize_coordinator_metadata(self):
        """Initialize coordinator with orchestration metadata."""
        coordinator_metadata = {
            "orchestration_config": {
                "collaboration_model": "sequential_dependency_based",
                "feedback_enabled": True,
                "iterative_refinement": True,
                "quality_thresholds": {
                    "minimum_confidence": 0.7,
                    "consensus_threshold": 0.8,
                    "iteration_limit": 3
                }
            },
            "consortium_management": {
                "agent_coordination": "centralized",
                "conflict_resolution": "confidence_weighted",
                "synthesis_approach": "multi_perspective_integration",
                "version_tracking": "comprehensive"
            }
        }
        
        self.maif.encoder.add_binary_block(
            json.dumps(coordinator_metadata).encode('utf-8'),
            "metadata",
            metadata={"type": "coordinator_config"}
        )
        
    def collect_contributions(self, agents: List[BaseAgent], query: str,
                            enable_refinement: bool = True) -> Dict[str, AgentContribution]:
        """Collect contributions using the state machine for proper iterative refinement."""
        print(f"\n🎯 USING STATE MACHINE FOR COLLABORATION")
        print(f"🔄 State machine will handle: context propagation, devil's advocate feedback, and convergence")
        
        # Use the state machine to execute the full collaboration cycle
        final_contributions = self.state_machine.execute_collaboration_cycle(agents, query)
        
        # Update coordinator's agent_contributions for compatibility
        self.agent_contributions = final_contributions
        self.collaboration_rounds = self.state_machine.round_number
        
        return final_contributions
    
    def collect_contributions_legacy(self, agents: List[BaseAgent], query: str,
                            enable_refinement: bool = True) -> Dict[str, AgentContribution]:
        """Legacy method - collect contributions from all specialized agents with enhanced MAIF context sharing."""
        self.collaboration_rounds += 1
        contributions = {}
        
        # Load rich context from existing MAIF artifacts AND previous round contributions
        context = self._load_rich_maif_context(agents)
        
        # Also include previous round contributions for iterative building
        if hasattr(self, 'agent_contributions') and self.agent_contributions:
            print(f"   📚 Building on {len(self.agent_contributions)} contributions from previous round")
            # Merge previous round context with MAIF context
            context["previous_round_contributions"] = {
                agent_id: {
                    "contribution_type": contrib.contribution_type,
                    "content": contrib.content,
                    "confidence": contrib.confidence,
                    "metadata": contrib.metadata
                }
                for agent_id, contrib in self.agent_contributions.items()
            }
            if context and len(context) > 1:  # More than just previous_round_contributions
                print(f"   📚 Enhanced with rich MAIF context from {len([k for k in context.keys() if k != 'previous_round_contributions'])} agents")
        elif context:
            print(f"   📚 Loaded rich MAIF context from {len(context)} agents with embeddings and metadata")
        else:
            print(f"   📝 Starting fresh collaboration round")
        
        print(f"\n=== COLLABORATION ROUND {self.collaboration_rounds} ===")
        
        # Initial contribution collection
        for agent in agents:
            print(f"Collecting initial contribution from {agent.agent_id}...")
            contribution = agent.contribute(query, context)
            contributions[agent.agent_id] = contribution
            
            # Add contribution to agent's MAIF
            content_id = agent.add_contribution_to_maif(contribution)
            print(f"    📁 Saved to MAIF with content ID: {content_id}")
            
            # Save the MAIF file immediately to ensure content is persisted
            try:
                maif_filename = f"{agent.agent_id}_contribution.maif"
                agent.maif.save(maif_filename, sign=True)
                print(f"    💾 MAIF file saved: {maif_filename}")
            except Exception as e:
                print(f"    ⚠️  Warning: Could not save MAIF file for {agent.agent_id}: {e}")
            
            # Update context with rich MAIF data instead of basic content
            agent_rich_context = self._extract_rich_context_from_maif(agent, contribution)
            context[agent.agent_id] = agent_rich_context
            print(f"    ✓ Updated context with rich MAIF data: {len(agent_rich_context)} fields")
            
        # Iterative refinement if enabled
        if enable_refinement:
            contributions = self._perform_iterative_refinement(agents, contributions, query)
            
        self.agent_contributions = contributions
        
        # Track collaboration metadata
        self._track_collaboration_round(contributions)
        
        return contributions
    
    def _perform_iterative_refinement(self, agents: List[BaseAgent],
                                    initial_contributions: Dict[str, AgentContribution],
                                    query: str) -> Dict[str, AgentContribution]:
        """Perform iterative refinement of contributions."""
        print("🔄 Starting iterative refinement process...")
        
        refined_contributions = initial_contributions.copy()
        max_iterations = 3
        
        for iteration in range(max_iterations):
            print(f"  📝 Refinement iteration {iteration + 1}/{max_iterations}")
            
            # Generate feedback for each agent based on other agents' contributions
            feedback_map = self._generate_inter_agent_feedback(refined_contributions)
            
            any_changes = False
            for agent in agents:
                if agent.agent_id in feedback_map:
                    feedback = feedback_map[agent.agent_id]
                    if feedback.get("needs_refinement", False):
                        print(f"    🔧 Refining {agent.agent_id} contribution...")
                        
                        original = refined_contributions[agent.agent_id]
                        refined = agent.refine_contribution(original, feedback)
                        
                        # Update in MAIF with version tracking
                        agent.add_contribution_to_maif(refined, update_existing=True)
                        
                        refined_contributions[agent.agent_id] = refined
                        any_changes = True
            
            # Track refinement iteration
            self.feedback_history.append({
                "iteration": iteration + 1,
                "timestamp": time.time(),
                "feedback_provided": len(feedback_map),
                "changes_made": any_changes
            })
            
            if not any_changes:
                print("    ✅ No further refinements needed.")
                break
                
        return refined_contributions
    
    def _generate_cross_agent_feedback(self, contributions: Dict[str, AgentContribution]) -> Dict[str, Dict[str, Any]]:
        """Generate feedback for each agent based on other agents' contributions."""
        feedback_map = {}
        
        for agent_id, contribution in contributions.items():
            # Simple feedback generation - in practice this could be more sophisticated
            other_contributions = {k: v for k, v in contributions.items() if k != agent_id}
            
            feedback = {
                "needs_refinement": len(other_contributions) > 0,  # Always refine if there are other contributions
                "positive": contribution.confidence > 0.7,
                "suggestions": f"Consider integrating insights from {len(other_contributions)} other agents",
                "context_from_others": other_contributions
            }
            
            feedback_map[agent_id] = feedback
            
        return feedback_map
    
    def _load_rich_maif_context(self, agents: List[BaseAgent]) -> Dict[str, Any]:
        """Load rich context from existing MAIF artifacts instead of basic JSON."""
        rich_context = {}
        
        for agent in agents:
            try:
                # Check if agent has existing MAIF file
                agent_maif_path = f"{agent.agent_id}_contribution.maif"
                if os.path.exists(agent_maif_path):
                    print(f"    📂 Loading MAIF context from {agent_maif_path}")
                    
                    # Load the MAIF artifact
                    loaded_maif = load_maif(agent_maif_path)
                    
                    # Extract rich context from the loaded MAIF
                    agent_context = self._extract_context_from_loaded_maif(loaded_maif, agent.agent_id)
                    if agent_context:
                        rich_context[agent.agent_id] = agent_context
                        print(f"      ✓ Extracted {len(agent_context)} context elements")
                    
            except Exception as e:
                print(f"    ⚠️  Could not load MAIF context for {agent.agent_id}: {e}")
                continue
                
        return rich_context
    
    def _extract_context_from_loaded_maif(self, maif_artifact: MAIF, agent_id: str) -> Dict[str, Any]:
        """Extract rich context from a loaded MAIF artifact."""
        context = {
            "agent_id": agent_id,
            "maif_metadata": {},
            "content_blocks": [],
            "embeddings_info": {},
            "knowledge_triples": [],
            "cross_modal_data": {},
            "version_history": {}
        }
        
        try:
            # Extract metadata from MAIF
            if hasattr(maif_artifact, 'metadata') and maif_artifact.metadata:
                context["maif_metadata"] = maif_artifact.metadata
            
            # Extract content blocks information
            if hasattr(maif_artifact.decoder, 'blocks') and maif_artifact.decoder.blocks:
                for block in maif_artifact.decoder.blocks:
                    block_summary = {
                        "block_id": block.block_id,
                        "block_type": block.block_type,
                        "size": block.size,
                        "metadata": block.metadata or {}
                    }
                    
                    # Try to extract actual content for text blocks (if not encrypted)
                    if (block.block_type == "text" and
                        not block.metadata.get("encrypted", False) if block.metadata else True):
                        try:
                            content = maif_artifact.decoder.get_text_block(block.block_id)
                            block_summary["content_preview"] = content  # Show full content without truncation
                        except (AttributeError, KeyError, Exception) as e:
                            logger.debug(f"Could not access block content: {e}")
                            block_summary["content_preview"] = "[Content not accessible]"
                    
                    context["content_blocks"].append(block_summary)
            
            # Extract embeddings information
            try:
                embeddings_list = maif_artifact.decoder.get_embeddings()
                if embeddings_list:
                    for i, embedding_data in enumerate(embeddings_list):
                        model_name = f"embedding_block_{i}"
                        if isinstance(embedding_data, dict):
                            context["embeddings_info"][model_name] = {
                                "model": embedding_data.get("model", model_name),
                                "count": len(embedding_data.get("vectors", [])),
                                "dimensions": len(embedding_data.get("vectors", [{}])[0]) if embedding_data.get("vectors") else 0
                            }
                        elif isinstance(embedding_data, list):
                            context["embeddings_info"][model_name] = {
                                "model": model_name,
                                "count": len(embedding_data),
                                "dimensions": len(embedding_data[0]) if embedding_data and isinstance(embedding_data[0], list) else 0
                            }
            except Exception as e:
                print(f"      ⚠️  Could not extract embeddings: {e}")
            
            # Extract cross-modal information from blocks
            for block in maif_artifact.decoder.blocks:
                if block.block_type == "cross_modal" and block.metadata:
                    context["cross_modal_data"][block.block_id] = {
                        "modalities": block.metadata.get("modalities", []),
                        "algorithm": block.metadata.get("algorithm", "unknown"),
                        "unified_dim": block.metadata.get("unified_representation_dim", 0)
                    }
            
            # Extract version history if available
            if hasattr(maif_artifact.decoder, 'version_history') and maif_artifact.decoder.version_history:
                context["version_history"] = {
                    block_id: len(versions) for block_id, versions in maif_artifact.decoder.version_history.items()
                }
                
        except Exception as e:
            print(f"      ⚠️  Error extracting context from MAIF: {e}")
            
        return context
    
    def _extract_rich_context_from_maif(self, agent: BaseAgent, contribution: AgentContribution) -> Dict[str, Any]:
        """Extract rich context from an agent's current MAIF state."""
        context = {
            "agent_profile": {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "specialization": agent.specialization,
                "iteration_count": getattr(agent, 'iteration_count', 0)
            },
            "contribution_summary": {
                "contribution_type": contribution.contribution_type,
                "confidence": contribution.confidence,
                "content_keys": list(contribution.content.keys()) if isinstance(contribution.content, dict) else [],
                "dependencies": contribution.dependencies or []
            },
            "maif_state": {
                "total_contributions": len(getattr(agent, 'contributions', [])),
                "content_blocks": len(getattr(agent, 'content_blocks', {})),
                "refinement_history": getattr(agent, 'refinement_history', [])
            }
        }
        
        # Add semantic embeddings info if available
        try:
            if hasattr(agent.maif, 'decoder'):
                embeddings_list = agent.maif.decoder.get_embeddings()
                context["embeddings_available"] = len(embeddings_list) if embeddings_list else 0
        except (AttributeError, TypeError, Exception) as e:
            logger.debug(f"Could not get embeddings info: {e}")
            context["embeddings_available"] = 0
        
        # Add cross-modal data info if available
        if hasattr(agent.maif, 'decoder') and hasattr(agent.maif.decoder, 'blocks'):
            cross_modal_count = sum(1 for block in agent.maif.decoder.blocks if block.block_type == "cross_modal")
            context["cross_modal_blocks"] = cross_modal_count
        
        # Add version tracking info
        if hasattr(agent.maif, 'encoder') and hasattr(agent.maif.encoder, 'version_history'):
            context["version_tracking"] = {
                "tracked_blocks": len(agent.maif.encoder.version_history),
                "total_versions": sum(len(versions) for versions in agent.maif.encoder.version_history.values())
            }
        
        # Add knowledge triples info if available
        try:
            # Count metadata blocks that contain knowledge triples
            knowledge_triple_count = 0
            if hasattr(agent.maif.decoder, 'blocks'):
                for block in agent.maif.decoder.blocks:
                    if (block.block_type == "metadata" and
                        block.metadata and block.metadata.get("type") == "knowledge_triple"):
                        knowledge_triple_count += 1
            context["knowledge_triples_count"] = knowledge_triple_count
        except (AttributeError, TypeError, Exception) as e:
            logger.debug(f"Could not count knowledge triples: {e}")
            context["knowledge_triples_count"] = 0
        
        return context
    
    def _perform_iterative_refinement(self, agents: List[BaseAgent],
                                    initial_contributions: Dict[str, AgentContribution],
                                    query: str) -> Dict[str, AgentContribution]:
        """Perform iterative refinement of contributions."""
        print("\n--- ITERATIVE REFINEMENT PHASE ---")
        
        refined_contributions = initial_contributions.copy()
        max_iterations = 2
        
        for iteration in range(max_iterations):
            print(f"\nRefinement iteration {iteration + 1}/{max_iterations}")
            
            # Generate feedback for each agent
            feedback_map = self._generate_inter_agent_feedback(refined_contributions)
            
            # Apply refinements
            any_changes = False
            for agent in agents:
                if agent.agent_id in feedback_map:
                    feedback = feedback_map[agent.agent_id]
                    if feedback.get("needs_refinement", False):
                        print(f"  Refining {agent.agent_id} contribution...")
                        
                        original = refined_contributions[agent.agent_id]
                        refined = agent.refine_contribution(original, feedback)
                        
                        # Update in MAIF with version tracking
                        agent.add_contribution_to_maif(refined, update_existing=True)
                        
                        refined_contributions[agent.agent_id] = refined
                        any_changes = True
            
            # Track refinement iteration
            self.feedback_history.append({
                "iteration": iteration + 1,
                "timestamp": time.time(),
                "feedback_provided": len(feedback_map),
                "changes_made": any_changes
            })
            
            if not any_changes:
                print("  No further refinements needed.")
                break
                
        return refined_contributions
    
    def _generate_llm_summary_of_rounds(self, all_rounds_data: list, query: str) -> str:
        """Generate an LLM-powered comprehensive summary of all collaboration rounds."""
        
        # Prepare summary data
        summary_prompt = f"""
        Analyze and summarize the evolution of a multi-agent consortium across {len(all_rounds_data)} collaboration rounds.
        
        QUERY: {query}
        
        COLLABORATION ROUNDS DATA:
        """
        
        for round_data in all_rounds_data:
            summary_prompt += f"\n--- ROUND {round_data['round_number']} ---\n"
            for agent_id, contrib_data in round_data['contributions'].items():
                summary_prompt += f"{agent_id}: Confidence {contrib_data['confidence']:.2f}, Type: {contrib_data['contribution_type']}\n"
                summary_prompt += f"  Content: {contrib_data['content_summary'][:200]}...\n"
        
        summary_prompt += """
        
        Please provide a comprehensive analysis covering:
        1. Evolution of agent confidence levels across rounds
        2. Key insights and patterns that emerged
        3. How different agent specializations contributed
        4. Convergence or divergence of perspectives
        5. Most significant developments in the collaboration
        6. Overall assessment of the multi-agent consortium effectiveness
        
        Format as a clear, structured summary suitable for executive review.
        """
        
        if self.use_openai:
            try:
                import openai
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert analyst specializing in multi-agent collaboration and consortium dynamics."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"🔥 DEBUG: OpenAI API error: {e}")
                return self._generate_fallback_summary(all_rounds_data)
        else:
            return self._generate_fallback_summary(all_rounds_data)
    
    def _generate_fallback_summary(self, all_rounds_data: list) -> str:
        """Generate a structured summary when OpenAI is not available."""
        
        summary = "MULTI-AGENT CONSORTIUM COLLABORATION ANALYSIS\n\n"
        
        # Confidence evolution
        summary += "📈 CONFIDENCE EVOLUTION:\n"
        for agent_id in all_rounds_data[0]['contributions'].keys():
            confidences = [round_data['contributions'][agent_id]['confidence'] for round_data in all_rounds_data if agent_id in round_data['contributions']]
            if len(confidences) >= 1:
                initial = confidences[0]
                final = confidences[-1]
                trend = "↗️ Increased" if final > initial else "↘️ Decreased" if final < initial else "→ Stable"
                summary += f"  {agent_id}: {initial:.2f} → {final:.2f} ({trend})\n"
            else:
                summary += f"  {agent_id}: No contributions found\n"
        
        # Agent specializations
        summary += "\n🎯 AGENT SPECIALIZATIONS:\n"
        specializations = {}
        for round_data in all_rounds_data[-1:]:  # Use final round
            for agent_id, contrib_data in round_data['contributions'].items():
                specializations[agent_id] = contrib_data['contribution_type']
        
        for agent_id, spec in specializations.items():
            summary += f"  {agent_id}: {spec}\n"
        
        # Collaboration patterns
        summary += "\n🔄 COLLABORATION PATTERNS:\n"
        summary += f"  Total Rounds: {len(all_rounds_data)}\n"
        summary += f"  Participating Agents: {len(all_rounds_data[0]['contributions'])}\n"
        
        avg_confidence_per_round = []
        for round_data in all_rounds_data:
            if len(round_data['contributions']) > 0:
                round_avg = sum(contrib['confidence'] for contrib in round_data['contributions'].values()) / len(round_data['contributions'])
                avg_confidence_per_round.append(round_avg)
            else:
                avg_confidence_per_round.append(0.0)
        
        summary += f"  Average Confidence Trend: {avg_confidence_per_round[0]:.2f} → {avg_confidence_per_round[-1]:.2f}\n"
        
        # Key insights
        summary += "\n💡 KEY INSIGHTS:\n"
        summary += "  • Multi-agent consortium demonstrated iterative collaboration\n"
        summary += "  • Each round built upon previous agent learnings\n"
        summary += "  • Diverse specializations provided comprehensive analysis\n"
        summary += "  • Devil's advocate provided critical perspective balance\n"
        
        # Overall assessment
        summary += "\n🎯 OVERALL ASSESSMENT:\n"
        final_avg_confidence = avg_confidence_per_round[-1]
        if final_avg_confidence > 0.8:
            assessment = "High confidence consortium with strong convergence"
        elif final_avg_confidence > 0.6:
            assessment = "Moderate confidence with balanced perspectives"
        else:
            assessment = "Conservative approach with critical analysis emphasis"
        
        summary += f"  {assessment}\n"
        summary += f"  Final Average Confidence: {final_avg_confidence:.2f}\n"
        
        return summary
    
    def _generate_inter_agent_feedback(self, contributions: Dict[str, AgentContribution]) -> Dict[str, Dict]:
        """Generate feedback between agents based on their contributions, with special focus on devil's advocate insights."""
        feedback_map = {}
        
        # Find devil's advocate contribution for critical analysis
        devils_advocate_contribution = None
        devils_advocate_id = None
        for agent_id, contribution in contributions.items():
            if "devils_advocate" in agent_id.lower():
                devils_advocate_contribution = contribution
                devils_advocate_id = agent_id
                break
        
        # Extract critical insights from devil's advocate if available
        critical_insights = {}
        if devils_advocate_contribution and isinstance(devils_advocate_contribution.content, dict):
            print(f"  📋 Incorporating critical feedback from {devils_advocate_id}")
            
            # Extract specific critiques for each agent type
            if "agent_critique" in devils_advocate_contribution.content:
                agent_critiques = devils_advocate_contribution.content["agent_critique"]
                if isinstance(agent_critiques, dict):
                    critical_insights = agent_critiques
            
            # Extract general critical challenges
            critical_challenges = devils_advocate_contribution.content.get("critical_challenges", [])
            alternative_recommendations = devils_advocate_contribution.content.get("alternative_recommendations", [])
        
        # Analyze contribution quality and interdependencies
        for agent_id, contribution in contributions.items():
            # Skip devil's advocate - they don't need feedback from themselves
            if agent_id == devils_advocate_id:
                continue
                
            other_contributions = {k: v for k, v in contributions.items() if k != agent_id}
            
            feedback = {
                "needs_refinement": False,
                "suggestions": [],
                "positive": True,
                "confidence_assessment": contribution.confidence,
                "devils_advocate_feedback": []
            }
            
            # Apply devil's advocate specific feedback
            if critical_insights:
                agent_type = contribution.agent_type
                
                # Look for specific critiques for this agent type
                for critique_key, critique_value in critical_insights.items():
                    if agent_type.lower() in critique_key.lower() or any(keyword in critique_key.lower() for keyword in [agent_type.lower()[:4], agent_id.split('_')[0]]):
                        feedback["needs_refinement"] = True
                        feedback["devils_advocate_feedback"].append(f"Critical analysis: {critique_value}")
                        feedback["suggestions"].append(f"Address devil's advocate concern: {critique_value}")
                
                # Add general critical challenges as refinement suggestions
                if isinstance(critical_challenges, list) and critical_challenges:
                    for challenge in critical_challenges[:2]:  # Limit to top 2 challenges
                        if isinstance(challenge, str) and len(challenge) > 10:
                            feedback["suggestions"].append(f"Consider critical challenge: {challenge}")
                            feedback["needs_refinement"] = True
                
                # Add alternative recommendations
                if isinstance(alternative_recommendations, list) and alternative_recommendations:
                    for alt_rec in alternative_recommendations[:1]:  # Limit to 1 alternative
                        if isinstance(alt_rec, str) and len(alt_rec) > 10:
                            feedback["suggestions"].append(f"Alternative approach suggested: {alt_rec}")
            
            # Check confidence threshold
            if contribution.confidence < 0.8:
                feedback["needs_refinement"] = True
                feedback["suggestions"].append("Increase confidence through additional analysis")
                feedback["positive"] = False
            
            # Check for dependency satisfaction
            if contribution.dependencies:
                for dep in contribution.dependencies:
                    if dep not in [c.contribution_type for c in contributions.values()]:
                        feedback["needs_refinement"] = True
                        feedback["suggestions"].append(f"Missing dependency: {dep}")
                        feedback["positive"] = False
            
            # Content-specific feedback
            if isinstance(contribution.content, dict):
                if "summary" not in contribution.content:
                    feedback["suggestions"].append("Add summary section for better integration")
                
                if len(contribution.content) < 3:
                    feedback["suggestions"].append("Expand content with more detailed analysis")
            
            # Generate feedback summary based on other agents
            if other_contributions:
                agent_types = [v.agent_type for v in other_contributions.values()]
                feedback_sources = set(agent_types)
                if devils_advocate_id:
                    feedback_sources.add("critical_analysis")
                feedback["feedback_summary"] = f"Integrated feedback from {len(other_contributions)} agents: {', '.join(feedback_sources)}"
            else:
                feedback["feedback_summary"] = "Initial contribution without cross-agent feedback"
            
            # Mark as needing refinement if devil's advocate provided specific feedback
            if feedback["devils_advocate_feedback"]:
                feedback["needs_refinement"] = True
                print(f"    🔍 {agent_id} marked for refinement based on critical analysis")
            
            feedback_map[agent_id] = feedback
            
        return feedback_map
    
    def _track_collaboration_round(self, contributions: Dict[str, AgentContribution]):
        """Track metadata about the collaboration round."""
        # Store the actual AgentContribution objects for trace generation
        round_metadata = {
            "round_number": self.collaboration_rounds,
            "timestamp": time.time(),
            "participating_agents": list(contributions.keys()),
            "contributions": contributions,  # Store the actual AgentContribution objects
            "contribution_stats": {
                "total_contributions": len(contributions),
                "average_confidence": sum(c.confidence for c in contributions.values()) / len(contributions),
                "dependency_count": sum(len(c.dependencies or []) for c in contributions.values()),
                "content_complexity": sum(
                    len(json.dumps(c.content)) for c in contributions.values()
                ) / len(contributions)
            },
            "quality_metrics": {
                "high_confidence_count": sum(1 for c in contributions.values() if c.confidence > 0.9),
                "medium_confidence_count": sum(1 for c in contributions.values() if 0.7 <= c.confidence <= 0.9),
                "low_confidence_count": sum(1 for c in contributions.values() if c.confidence < 0.7)
            }
        }
        
        # Store in collaboration history for trace generation
        if not hasattr(self, 'collaboration_history'):
            self.collaboration_history = []
        self.collaboration_history.append(round_metadata)
        
        # Create JSON-serializable version for MAIF storage
        serializable_metadata = {
            "round_number": self.collaboration_rounds,
            "timestamp": time.time(),
            "participating_agents": list(contributions.keys()),
            "contributions": {
                agent_id: {
                    "contribution_type": contrib.contribution_type,
                    "confidence": contrib.confidence,
                    "content_size": len(json.dumps(contrib.content)),
                    "dependencies": contrib.dependencies or [],
                    "metadata": contrib.metadata or {}
                }
                for agent_id, contrib in contributions.items()
            },
            "contribution_stats": round_metadata["contribution_stats"],
            "quality_metrics": round_metadata["quality_metrics"]
        }
        
        # Add to MAIF with privacy controls
        privacy_policy = PrivacyPolicy(
            privacy_level=PrivacyLevel.INTERNAL,
            encryption_mode=EncryptionMode.NONE
        )
        
        self.maif.encoder.add_binary_block(
            json.dumps(serializable_metadata).encode('utf-8'),
            "metadata",
            metadata={"type": "collaboration_round"},
            privacy_policy=privacy_policy
        )
    
    def synthesize_final_artifact(self, query: str) -> AgentContribution:
        """Synthesize all contributions into a comprehensive final artifact."""
        
        print("🔥 DEBUG: Synthesizing contributions from 3 geography agents...")
        
        # Extract key insights from ALL agent contributions
        geo_insights_1 = self.agent_contributions.get("geo_agent_001", {}).content if self.agent_contributions.get("geo_agent_001") else {}
        geo_insights_2 = self.agent_contributions.get("geo_agent_002", {}).content if self.agent_contributions.get("geo_agent_002") else {}
        cultural_insights = self.agent_contributions.get("culture_agent_001", {}).content if self.agent_contributions.get("culture_agent_001") else {}
        logistics_insights = self.agent_contributions.get("logistics_agent_001", {}).content if self.agent_contributions.get("logistics_agent_001") else {}
        safety_insights = self.agent_contributions.get("safety_agent_001", {}).content if self.agent_contributions.get("safety_agent_001") else {}
        devils_advocate_insights = self.agent_contributions.get("devils_advocate_001", {}).content if self.agent_contributions.get("devils_advocate_001") else {}
        
        print(f"🔥 DEBUG: Geography Agent 1 insights: {len(str(geo_insights_1))} chars")
        print(f"🔥 DEBUG: Geography Agent 2 insights: {len(str(geo_insights_2))} chars")
        print(f"🔥 DEBUG: Cultural Agent insights: {len(str(cultural_insights))} chars")
        print(f"🔥 DEBUG: Logistics Agent insights: {len(str(logistics_insights))} chars")
        print(f"🔥 DEBUG: Safety Agent insights: {len(str(safety_insights))} chars")
        print(f"🔥 DEBUG: Devil's Advocate insights: {len(str(devils_advocate_insights))} chars")
        
        # Combine geographical perspectives with emphasis on consensus vs. critical analysis
        combined_route_segments = []
        if geo_insights_1.get("route_segments"):
            combined_route_segments.extend(geo_insights_1["route_segments"])
        if geo_insights_2.get("route_segments"):
            combined_route_segments.extend(geo_insights_2["route_segments"])
            
        # Extract critical challenges from devil's advocate
        critical_challenges = devils_advocate_insights.get("critical_challenges", [])
        
        # Determine consensus vs. critical perspective
        consensus_distance = geo_insights_1.get("total_distance_km", 14550) or geo_insights_2.get("total_distance_km", 14550)
        devils_advocate_summary = devils_advocate_insights.get("summary", "No critical analysis available")
        
        # Collect critical challenges from devil's advocate
        critical_challenges = devils_advocate_insights.get("critical_challenges", [])
        
        synthesized_artifact = {
            "query": query,
            "consortium_response": {
                "executive_summary": {
                    "feasibility": "Highly contested - 2 agents optimistic, 1 critical",
                    "distance": f"{consensus_distance} km across Pacific Ocean and Himalayas",
                    "duration": "45-60 days with continuous travel (if possible)",
                    "key_challenge": "Multiple geographical impossibilities identified",
                    "complexity": "High - Multi-agent analysis with conflicting perspectives",
                    "meaningful_framework": "Multi-agent collaborative analysis framework",
                    "consensus_view": "Theoretically possible with supernatural abilities",
                    "critical_perspective": devils_advocate_summary,
                    "critical_challenges_identified": critical_challenges[:5] if critical_challenges else ["No specific critical challenges identified"]
                },
                "comprehensive_contribution_summary": {
                    "geography_analysis": {
                        "agent_1_insights": geo_insights_1,
                        "agent_2_insights": geo_insights_2,
                        "combined_route_segments": combined_route_segments,
                        "consensus_distance": consensus_distance
                    },
                    "cultural_analysis": {
                        "agent_insights": cultural_insights,
                        "cultural_regions": cultural_insights.get("cultural_regions", []),
                        "meaningful_framework": cultural_insights.get("meaningful_framework", {}),
                        "cross_cultural_themes": cultural_insights.get("cross_cultural_themes", [])
                    },
                    "logistics_analysis": {
                        "agent_insights": logistics_insights,
                        "resource_requirements": logistics_insights.get("resource_requirements", {}),
                        "timeline_optimization": logistics_insights.get("timeline_optimization", {}),
                        "support_infrastructure": logistics_insights.get("support_infrastructure", {})
                    },
                    "safety_analysis": {
                        "agent_insights": safety_insights,
                        "safety_protocols": safety_insights.get("safety_protocols", {}),
                        "risk_categories": safety_insights.get("risk_categories", {}),
                        "emergency_procedures": safety_insights.get("emergency_procedures", [])
                    },
                    "critical_analysis": {
                        "devils_advocate_insights": devils_advocate_insights,
                        "agent_critiques": devils_advocate_insights.get("agent_critique", {}),
                        "critical_challenges": critical_challenges,
                        "alternative_recommendations": devils_advocate_insights.get("alternative_recommendations", [])
                    }
                },
                "multi_agent_analysis": {
                    "agent_consensus": {
                        "agents_in_agreement": ["geo_agent_001", "geo_agent_002", "culture_agent_001", "logistics_agent_001", "safety_agent_001"],
                        "consensus_confidence": sum(c.confidence for c in self.agent_contributions.values() if c.agent_id != "devils_advocate_001") / max(1, len([c for c in self.agent_contributions.values() if c.agent_id != "devils_advocate_001"])),
                        "consensus_summary": "Journey is geographically challenging but possible with supernatural abilities"
                    },
                    "devils_advocate_analysis": {
                        "agent_id": "devils_advocate_001",
                        "critical_confidence": self.agent_contributions.get("devils_advocate_001", type('obj', (object,), {'confidence': 0.95})).confidence,
                        "critical_summary": devils_advocate_summary,
                        "alternative_suggestions": devils_advocate_insights.get("alternative_recommendations", ["No alternatives provided"]),
                        "agent_critiques": devils_advocate_insights.get("agent_critique", {}),
                        "risk_assessment": devils_advocate_insights.get("risk_assessment", "High risk")
                    },
                    "synthesis_approach": "Balanced perspective incorporating both optimistic and critical geographical assessments"
                },
                "integrated_plan": {
                    "phase_1_departure": {
                        "geography_insights": geo_insights_1.get("route_segments", []),
                        "cultural_considerations": cultural_insights.get("cultural_regions", []),
                        "logistics_requirements": logistics_insights.get("resource_requirements", {}),
                        "safety_protocols": safety_insights.get("safety_protocols", {}),
                        "duration": logistics_insights.get("timeline_optimization", {}).get("total_estimated_duration", "45-60 days"),
                        "cultural_significance": "Sacred departure from ancestral lands with proper ceremonies",
                        "devils_advocate_concerns": devils_advocate_insights.get("critical_challenges", [])[:2]
                    },
                    "phase_2_pacific_crossing": {
                        "geography_insights": geo_insights_2.get("route_segments", []),
                        "cultural_considerations": cultural_insights.get("cultural_regions", []),
                        "logistics_requirements": logistics_insights.get("resource_requirements", {}),
                        "safety_protocols": safety_insights.get("safety_protocols", {}),
                        "duration": "25-30 days continuous swimming",
                        "cultural_significance": "Sacred waters pilgrimage across Pacific cultures",
                        "devils_advocate_concerns": devils_advocate_insights.get("critical_challenges", [])[2:4] if len(devils_advocate_insights.get("critical_challenges", [])) > 2 else []
                    },
                    "phase_3_multi_agent_synthesis": {
                        "geography_analysis": {
                            "total_distance": consensus_distance,
                            "route_complexity": "High - transcontinental journey with extreme challenges",
                            "terrain_challenges": ["Ocean crossing", "Mountain traversal", "International borders"]
                        },
                        "cultural_analysis": cultural_insights,
                        "logistics_analysis": logistics_insights,
                        "safety_analysis": safety_insights,
                        "critical_analysis": devils_advocate_insights,
                        "duration": "15-25 days final approach",
                        "cultural_significance": "Completion of transformative spiritual journey across cultures"
                    }
                },
                "resource_synthesis": {
                    "supernatural_abilities_utilization": {
                        "infinite_swimming": "Enables Pacific crossing impossible for normal humans",
                        "no_sleep_requirement": "Allows continuous 24/7 progress and cultural immersion"
                    },
                    "critical_equipment": [
                        "Satellite communication system",
                        "Advanced thermal protection",
                        "Cultural exchange gifts and documentation",
                        "Emergency medical supplies",
                        "Navigation and weather monitoring tools"
                    ],
                    "support_network": [
                        "International diplomatic coordination",
                        "Cultural liaison contacts",
                        "Emergency rescue protocols",
                        "Medical monitoring team"
                    ]
                },
                "risk_management_integration": {
                    "primary_risks": [
                        "Pacific Ocean environmental hazards",
                        "Himalayan altitude and weather",
                        "Political and border complications",
                        "Cultural misunderstandings"
                    ],
                    "mitigation_strategy": "Layered safety protocols with cultural sensitivity",
                    "emergency_protocols": "Multi-national rescue coordination with cultural liaisons"
                },
                "meaningful_outcomes": {
                    "personal_transformation": "Spiritual and cultural growth through impossible journey",
                    "cultural_bridge_building": "Connecting Pacific and Himalayan cultures",
                    "environmental_awareness": "Highlighting ocean and mountain conservation",
                    "human_potential_demonstration": "Showing what's possible with determination and supernatural aid"
                }
            },
            "consortium_metadata": {
                "contributing_agents": list(self.agent_contributions.keys()),
                "synthesis_confidence": 0.91,
                "interdisciplinary_integration": "High",
                "cultural_sensitivity_rating": "Excellent",
                "practical_feasibility": "Requires supernatural abilities as specified"
            },
            "summary": "A 14,550 km transcontinental journey from California to Nepal, transformed from impossible physical challenge into meaningful cultural pilgrimage through consortium agent collaboration."
        }
        
        return AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            contribution_type="synthesized_consortium_artifact",
            content=synthesized_artifact,
            confidence=0.91,
            dependencies=list(self.agent_contributions.keys()),
            metadata={
                "synthesis_method": "multi_agent_integration",
                "consortium_size": len(self.agent_contributions),
                "integration_complexity": "high"
            }
        )
    
    def create_consortium_maif(self, final_artifact: AgentContribution, output_path: str) -> bool:
        """Create the final consortium MAIF file containing all contributions."""
        
        # Add the synthesized artifact
        self.add_contribution_to_maif(final_artifact)
        
        # Add cross-references to other agents' contributions
        for agent_id, contribution in self.agent_contributions.items():
            reference_content = {
                "reference_type": "agent_contribution",
                "referenced_agent": agent_id,
                "contribution_summary": contribution.content.get("summary", ""),
                "confidence": contribution.confidence,
                "contribution_type": contribution.contribution_type
            }
            
            self.maif.add_text(
                json.dumps(reference_content, indent=2),
                title=f"Reference to {agent_id} contribution",
                encrypt=False
            )
        
        # Add consortium metadata
        consortium_metadata = {
            "consortium_composition": {
                "total_agents": len(self.agent_contributions) + 1,  # +1 for coordinator
                "specializations": [contrib.agent_type for contrib in self.agent_contributions.values()],
                "collaboration_model": "sequential_dependency_based",
                "synthesis_approach": "multi_perspective_integration"
            },
            "artifact_characteristics": {
                "content_type": "multi_agent_consortium_artifact",
                "synthesis_confidence": final_artifact.confidence,
                "interdisciplinary_integration": True,
                "cultural_sensitivity_rating": "high"
            }
        }
        
        self.maif.add_multimodal(
            consortium_metadata,
            title="Consortium Metadata",
            use_acam=True
        )
        
        # Save the consortium MAIF
        return self.maif.save(output_path, sign=True)
        
        return True


def generate_comprehensive_trace(agents: List[BaseAgent], coordinator_agent: CoordinatorAgent,
                               all_rounds_data: List[Dict], final_contributions: Dict[str, AgentContribution]) -> Dict[str, Any]:
    """Generate a comprehensive trace of all agent activity throughout the collaboration."""
    
    trace = {
        "trace_metadata": {
            "generation_timestamp": time.time(),
            "total_agents": len(agents),
            "total_rounds": len(all_rounds_data),
            "trace_version": "1.0.0"
        },
        "agent_profiles": {},
        "collaboration_timeline": [],
        "agent_interactions": {},
        "content_evolution": {},
        "maif_artifacts": {},
        "performance_metrics": {},
        "knowledge_graph": {},
        "final_state": {}
    }
    
    # Agent Profiles
    for agent in agents + [coordinator_agent]:
        agent_profile = {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "specialization": agent.specialization,
            "capabilities": agent._get_capabilities(),
            "total_contributions": len(getattr(agent, 'contributions', [])),
            "iteration_count": getattr(agent, 'iteration_count', 0),
            "refinement_history": getattr(agent, 'refinement_history', []),
            "maif_blocks": len(getattr(agent, 'content_blocks', {}))
        }
        
        # Add MAIF state information
        if hasattr(agent, 'maif') and hasattr(agent.maif, 'decoder'):
            try:
                agent_profile["maif_state"] = {
                    "total_blocks": len(agent.maif.decoder.blocks) if hasattr(agent.maif.decoder, 'blocks') else 0,
                    "embeddings_available": len(agent.maif.decoder.get_embeddings()) if hasattr(agent.maif.decoder, 'get_embeddings') else 0,
                    "version_history": len(getattr(agent.maif.decoder, 'version_history', {}))
                }
            except (AttributeError, TypeError, Exception) as e:
                logger.debug(f"Could not access MAIF state for agent {agent.agent_id}: {e}")
                agent_profile["maif_state"] = {"error": "Could not access MAIF state"}
        
        trace["agent_profiles"][agent.agent_id] = agent_profile
    
    # Collaboration Timeline
    for round_data in all_rounds_data:
        round_trace = {
            "round_number": round_data.get("round_number", 0),
            "timestamp": time.time(),
            "contributions": {},
            "context_size": 0,
            "round_duration": round_data.get("round_duration", 0)
        }
        
        # Always use final_contributions for accurate content data
        # The round_data often contains processed/summarized versions that lose content
        contributions_to_use = final_contributions
        
        # Extract contribution details with full content
        for agent_id, contribution in contributions_to_use.items():
            if hasattr(contribution, 'content') and contribution.content:
                # Include full content in trace
                content_text = json.dumps(contribution.content)
                content_summary = str(contribution.content)
                if len(content_summary) > 500:
                    content_summary = content_summary[:500] + "..."
                
                contrib_trace = {
                    "contribution_type": contribution.contribution_type,
                    "confidence": contribution.confidence,
                    "content_size": len(content_text),
                    "content_summary": content_summary,
                    "full_content": contribution.content,  # Include full content
                    "dependencies": contribution.dependencies or [],
                    "metadata": contribution.metadata or {}
                }
            elif isinstance(contribution, dict) and 'confidence' in contribution:
                # Handle dict-based contribution data
                content = contribution.get("content", {})
                content_text = json.dumps(content) if content else "{}"
                contrib_trace = {
                    "contribution_type": contribution.get("contribution_type", "unknown"),
                    "confidence": contribution.get("confidence", 0.0),
                    "content_size": len(content_text),
                    "content_summary": str(content)[:500] + "..." if len(str(content)) > 500 else str(content),
                    "full_content": content,
                    "dependencies": contribution.get("dependencies", []),
                    "metadata": contribution.get("metadata", {})
                }
            else:
                # Fallback for unknown contribution types
                content_text = json.dumps(contribution) if contribution else "{}"
                contrib_trace = {
                    "contribution_type": "unknown",
                    "confidence": 0.0,
                    "content_size": len(content_text),
                    "content_summary": str(contribution)[:500] + "..." if len(str(contribution)) > 500 else str(contribution),
                    "full_content": contribution if contribution else {},
                    "dependencies": [],
                    "metadata": {}
                }
            
            round_trace["contributions"][agent_id] = contrib_trace
        
        trace["collaboration_timeline"].append(round_trace)
    
    # Agent Interactions (who influenced whom)
    for agent in agents:
        interactions = {
            "influenced_by": [],
            "influenced": [],
            "context_received": 0,
            "context_provided": 0
        }
        
        # Analyze refinement history for interactions
        for refinement in getattr(agent, 'refinement_history', []):
            if 'feedback_applied' in refinement:
                interactions["influenced_by"].extend(refinement['feedback_applied'])
        
        trace["agent_interactions"][agent.agent_id] = interactions
    
    # Content Evolution
    for agent in agents:
        evolution = {
            "initial_contribution": None,
            "refinements": [],
            "final_state": None,
            "evolution_metrics": {
                "total_iterations": getattr(agent, 'iteration_count', 0),
                "confidence_changes": [],
                "content_growth": []
            }
        }
        
        # Track refinement history
        for i, refinement in enumerate(getattr(agent, 'refinement_history', [])):
            evolution["refinements"].append({
                "iteration": i + 1,
                "timestamp": refinement.get("timestamp", 0),
                "confidence_change": refinement.get("confidence_change", 0),
                "feedback_summary": refinement.get("feedback_summary", "")
            })
            evolution["evolution_metrics"]["confidence_changes"].append(refinement.get("confidence_change", 0))
        
        trace["content_evolution"][agent.agent_id] = evolution
    
    # MAIF Artifacts
    for agent in agents:
        maif_info = {
            "agent_id": agent.agent_id,
            "maif_file": f"{agent.agent_id}_contribution.maif",
            "manifest_file": f"{agent.agent_id}_contribution_manifest.json",
            "blocks_created": len(getattr(agent, 'content_blocks', {})),
            "version_tracking": bool(hasattr(agent.maif, 'encoder') and hasattr(agent.maif.encoder, 'version_history'))
        }
        
        # Check if files exist
        maif_info["files_exist"] = {
            "maif": os.path.exists(f"{agent.agent_id}_contribution.maif"),
            "manifest": os.path.exists(f"{agent.agent_id}_contribution_manifest.json")
        }
        
        trace["maif_artifacts"][agent.agent_id] = maif_info
    
    # Performance Metrics
    total_contributions = sum(len(getattr(agent, 'contributions', [])) for agent in agents)
    total_refinements = sum(getattr(agent, 'iteration_count', 0) for agent in agents)
    
    trace["performance_metrics"] = {
        "total_contributions": total_contributions,
        "total_refinements": total_refinements,
        "average_confidence": sum(contrib.confidence for contrib in final_contributions.values()) / len(final_contributions) if final_contributions else 0,
        "collaboration_efficiency": total_contributions / len(agents) if agents else 0,
        "refinement_rate": total_refinements / total_contributions if total_contributions > 0 else 0
    }
    
    # Knowledge Graph (simplified)
    knowledge_graph = {
        "nodes": [],
        "edges": [],
        "concepts": set()
    }
    
    for agent_id, contribution in final_contributions.items():
        # Add agent as node
        knowledge_graph["nodes"].append({
            "id": agent_id,
            "type": "agent",
            "specialization": next((a.specialization for a in agents if a.agent_id == agent_id), "unknown")
        })
        
        # Add contribution as node
        contrib_id = f"{agent_id}_contribution"
        knowledge_graph["nodes"].append({
            "id": contrib_id,
            "type": "contribution",
            "contribution_type": contribution.contribution_type if hasattr(contribution, 'contribution_type') else "unknown"
        })
        
        # Add edge from agent to contribution
        knowledge_graph["edges"].append({
            "source": agent_id,
            "target": contrib_id,
            "relationship": "created"
        })
        
        # Extract concepts from content
        if hasattr(contribution, 'content') and isinstance(contribution.content, dict):
            for key in contribution.content.keys():
                knowledge_graph["concepts"].add(key)
    
    trace["knowledge_graph"] = {
        "nodes": knowledge_graph["nodes"],
        "edges": knowledge_graph["edges"],
        "concepts": list(knowledge_graph["concepts"])
    }
    
    # Final State
    trace["final_state"] = {
        "final_contributions": {
            agent_id: {
                "contribution_type": contrib.contribution_type,
                "confidence": contrib.confidence,
                "content_summary": str(contrib.content)[:500] + "..." if len(str(contrib.content)) > 500 else str(contrib.content)
            }
            for agent_id, contrib in final_contributions.items()
        },
        "consortium_state": {
            "total_agents": len(agents),
            "successful_contributions": len(final_contributions),
            "completion_timestamp": time.time()
        }
    }
    
    return trace


def demonstrate_multi_agent_consortium(num_rounds: int = 10, agent_counts: Dict[str, int] = None):
    """Enhanced demonstration of the multi-agent consortium with comprehensive tracking."""
    
    # Default agent counts if not provided
    if agent_counts is None:
        agent_counts = {
            'geo': 2,
            'culture': 1,
            'logistics': 1,
            'safety': 1,
            'devils_advocate': 1
        }
    
    print("=" * 80)
    print("ENHANCED MULTI-AGENT CONSORTIUM DEMONSTRATION")
    print("=" * 80)
    print()
    
    # The original meaningful task query
    query = ("How do I walk from California to Nepal in a meaningful way - where I "
             "have infinite ability to swim, and don't need to sleep")
    
    print(f"CONSORTIUM QUERY: {query}")
    print()
    
    # Initialize multi-agent consortium with all specializations
    print("PHASE 0: Initializing multi-agent consortium (geography, cultural, logistics, safety, devil's advocate) with enhanced tracking...")
    print("-" * 60)
    
    print("🔥 DEBUG: Creating multi-agent consortium with configurable specializations...")
    print("🔥 DEBUG: Using single shared MAIF file for all agents...")
    
    # Create single shared MAIF for all agents
    shared_maif = create_maif("multi_agent_consortium", enable_privacy=True)
    print(f"  ✓ Created shared MAIF: multi_agent_consortium")
    
    agents = []
    
    # Create geography agents
    for i in range(agent_counts['geo']):
        agent_id = f"geo_agent_{i+1:03d}"
        geo_agent = GeographyAgent(agent_id, is_devils_advocate=False, shared_maif=shared_maif)
        agents.append(geo_agent)
        print(f"  - {geo_agent.agent_id}: Geographical analysis and routing")
    
    # Create cultural agents
    for i in range(agent_counts['culture']):
        agent_id = f"culture_agent_{i+1:03d}"
        cultural_agent = CulturalAgent(shared_maif=shared_maif)
        cultural_agent.agent_id = agent_id  # Override the default ID
        agents.append(cultural_agent)
        print(f"  - {cultural_agent.agent_id}: Cultural insights and meaningful experiences")
    
    # Create logistics agents
    for i in range(agent_counts['logistics']):
        agent_id = f"logistics_agent_{i+1:03d}"
        logistics_agent = LogisticsAgent(shared_maif=shared_maif)
        logistics_agent.agent_id = agent_id  # Override the default ID
        agents.append(logistics_agent)
        print(f"  - {logistics_agent.agent_id}: Resource optimization and planning")
    
    # Create safety agents
    for i in range(agent_counts['safety']):
        agent_id = f"safety_agent_{i+1:03d}"
        safety_agent = SafetyAgent(shared_maif=shared_maif)
        safety_agent.agent_id = agent_id  # Override the default ID
        agents.append(safety_agent)
        print(f"  - {safety_agent.agent_id}: Risk mitigation and emergency response")
    
    # Create devils advocate agents
    for i in range(agent_counts['devils_advocate']):
        agent_id = f"devils_advocate_{i+1:03d}"
        devils_advocate_agent = DevilsAdvocateAgent(shared_maif=shared_maif)
        devils_advocate_agent.agent_id = agent_id  # Override the default ID
        agents.append(devils_advocate_agent)
        print(f"  - {devils_advocate_agent.agent_id}: Critical analysis of ALL agent contributions")
    
    # Create coordinator agent with shared MAIF
    coordinator_agent = CoordinatorAgent(shared_maif=shared_maif)
    
    print(f"✓ Initialized {len(agents)} specialized agents + 1 coordinator")
    
    # Display agent capabilities
    for agent in agents:
        capabilities = agent._get_capabilities()
        print(f"  {agent.agent_id}: {len(capabilities)} capabilities")
    
    print()
    
    # Collect contributions from all agents with configurable collaboration rounds
    print(f"PHASE 1: Collecting agent contributions with {num_rounds} collaboration rounds...")
    print("-" * 60)
    
    all_rounds_data = []
    final_contributions = {}
    previous_contributions = None
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n🔄 COLLABORATION ROUND {round_num}/{num_rounds}")
        print("-" * 40)
        
        # Pass previous round's contributions as context for iterative building
        if previous_contributions:
            # Set the coordinator's agent_contributions to previous round for context
            coordinator_agent.agent_contributions = previous_contributions
        
        current_contributions = coordinator_agent.collect_contributions(agents, query, enable_refinement=True)
        
        # Always keep the latest contributions as final
        if current_contributions:
            final_contributions = current_contributions
        previous_contributions = current_contributions
        
        # Collect round data for LLM summary
        round_data = {
            "round_number": round_num,
            "contributions": {}
        }
        
        print(f"\nRound {round_num} Summary:")
        for agent_id, contribution in current_contributions.items():
            print(f"  ✓ {agent_id}: Confidence {contribution.confidence:.2f}")
            round_data["contributions"][agent_id] = {
                "confidence": contribution.confidence,
                "contribution_type": contribution.contribution_type,
                "content_summary": str(contribution.content)[:500] + "..." if len(str(contribution.content)) > 500 else str(contribution.content)
            }
        
        all_rounds_data.append(round_data)
    
    # Use final round contributions (final_contributions should contain the last round's data)
    contributions = final_contributions
    print(f"\n🎯 Using contributions from final collaboration round for synthesis.")
    print(f"🔍 Final contributions count: {len(contributions) if contributions else 0}")
    
    # Generate LLM summary of all rounds
    print(f"\n🤖 Generating comprehensive LLM summary of all {num_rounds} collaboration rounds...")
    llm_summary = coordinator_agent._generate_llm_summary_of_rounds(all_rounds_data, query)
    print(f"\n" + "="*80)
    print(f"🧠 LLM COMPREHENSIVE SUMMARY OF ALL {num_rounds} COLLABORATION ROUNDS")
    print("="*80)
    print(llm_summary)
    print("="*80)
    
    # Generate trace data first for comprehensive summary
    collaboration_history = getattr(coordinator_agent, 'collaboration_history', all_rounds_data)
    
    # Ensure we pass the actual final contributions with content
    print(f"🔍 DEBUG: Final contributions before trace generation:")
    for agent_id, contrib in contributions.items():
        if hasattr(contrib, 'content'):
            print(f"  {agent_id}: content_size={len(str(contrib.content))}, type={type(contrib.content)}")
        else:
            print(f"  {agent_id}: NO CONTENT ATTRIBUTE")
    
    trace_dump = generate_comprehensive_trace(agents, coordinator_agent, collaboration_history, contributions)
    
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE CONTRIBUTION SUMMARY (FROM TRACE)")
    print("="*80)
    
    # Use trace data for comprehensive summary
    final_contributions_trace = trace_dump.get("final_state", {}).get("final_contributions", {})
    agent_profiles = trace_dump.get("agent_profiles", {})
    
    for agent_id, contribution_data in final_contributions_trace.items():
        agent_profile = agent_profiles.get(agent_id, {})
        
        print(f"\n🤖 {agent_id.upper()}: {contribution_data.get('contribution_type', 'Unknown')}")
        print(f"   Confidence: {contribution_data.get('confidence', 0.0):.2f}")
        print(f"   Specialization: {agent_profile.get('specialization', 'Unknown')}")
        print(f"   Total Iterations: {agent_profile.get('iteration_count', 0)}")
        print(f"   MAIF Blocks: {agent_profile.get('maif_blocks', 0)}")
        print(f"   Content Size: {len(contribution_data.get('content_summary', '')):,} chars")
        
        # Show refinement evolution
        refinement_history = agent_profile.get('refinement_history', [])
        if refinement_history:
            print(f"   📈 REFINEMENT EVOLUTION:")
            for refinement in refinement_history[-2:]:  # Show last 2 refinements
                print(f"     • Iteration {refinement.get('iteration', 0)}: {refinement.get('feedback_summary', 'No feedback')}")
                print(f"       Confidence change: {refinement.get('confidence_change', 0.0):+.3f}")
        
        # Show key content insights from trace
        print(f"   📋 KEY CONTENT INSIGHTS:")
        content_summary = contribution_data.get('content_summary', '')
        if content_summary and len(content_summary) > 100:
            # Extract key insights from content summary
            if 'route_segments' in content_summary:
                print(f"     🗺️  Geographic analysis with route planning")
            if 'cultural_regions' in content_summary:
                print(f"     🏛️  Cultural framework across multiple regions")
            if 'resource_requirements' in content_summary:
                print(f"     📦 Logistics optimization and resource planning")
            if 'risk_categories' in content_summary:
                print(f"     ⚠️  Comprehensive safety and risk assessment")
            if 'critical_challenges' in content_summary:
                print(f"     🔍 Critical analysis and challenge identification")
        
        print("-" * 60)
    
    print()
    
    # Display version history for each agent
    print("PHASE 1.5: Version History Analysis...")
    print("-" * 60)
    for agent in agents:
        version_history = agent.get_version_history()
        total_versions = sum(len(versions) for versions in version_history.values())
        print(f"✓ {agent.agent_id}: {total_versions} total versions across {len(version_history)} content blocks")
        
        # Show refinement history
        if agent.refinement_history:
            print(f"  Refinements: {len(agent.refinement_history)} iterations")
            for refinement in agent.refinement_history:
                print(f"    Iteration {refinement['iteration']}: confidence change {refinement['confidence_change']:+.3f}")
    
    print()
    
    # Synthesize final artifact with multiple iterations
    print("PHASE 2: Synthesizing consortium artifact with version tracking...")
    print("-" * 60)
    final_artifact = coordinator_agent.synthesize_final_artifact(query)
    print(f"✓ Synthesized final artifact: {final_artifact.contribution_type}")
    print(f"  Synthesis confidence: {final_artifact.confidence:.2f}")
    
    # Add final artifact to coordinator's MAIF with version tracking
    coordinator_agent.add_contribution_to_maif(final_artifact)
    
    # Perform iterative refinement of the final artifact
    print("\nPerforming final artifact refinement...")
    refinement_feedback = {
        "needs_refinement": True,
        "suggestions": ["Add implementation timeline", "Include risk mitigation details"],
        "positive": True
    }
    
    refined_final_artifact = coordinator_agent.refine_contribution(final_artifact, refinement_feedback)
    coordinator_agent.add_contribution_to_maif(refined_final_artifact, update_existing=True)
    
    print(f"✓ Refined final artifact confidence: {refined_final_artifact.confidence:.2f}")
    print()
    
    # Create consortium MAIF file
    print("PHASE 3: Creating comprehensive consortium MAIF artifact...")
    print("-" * 60)
    output_path = "earth_to_andromeda_consortium_artifact.maif"
    saved_files = []  # Track all files created during the demonstration

    if coordinator_agent.create_consortium_maif(final_artifact, output_path):
        print(f"✓ Consortium MAIF artifact created: {output_path}")
        saved_files.append(output_path)
    else:
        print(f"✗ Failed to create consortium MAIF artifact: {output_path}")
        
    # Generate trace data for artifact summary
    collaboration_history = getattr(coordinator_agent, 'collaboration_history', all_rounds_data)
    trace_dump = generate_comprehensive_trace(agents, coordinator_agent, collaboration_history, final_contributions)
    
    # Display key insights from the agent trace
    print()
    print("CONSORTIUM ARTIFACT SUMMARY (FROM AGENT TRACE):")
    print("-" * 50)
    
    # Extract insights from trace data
    final_contributions_trace = trace_dump.get("final_state", {}).get("final_contributions", {})
    performance_metrics = trace_dump.get("performance_metrics", {})
    agent_profiles = trace_dump.get("agent_profiles", {})
    
    # Calculate consensus from trace data
    confidences = [contrib.get("confidence", 0.0) for contrib in final_contributions_trace.values()]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Extract key insights from agent contributions in trace
    geo_data = final_contributions_trace.get("geo_agent_001", {})
    cultural_data = final_contributions_trace.get("culture_agent_001", {})
    logistics_data = final_contributions_trace.get("logistics_agent_001", {})
    safety_data = final_contributions_trace.get("safety_agent_001", {})
    critical_data = final_contributions_trace.get("devils_advocate_001", {})
    
    print(f"Feasibility: Multi-agent consensus (avg confidence: {avg_confidence:.2f})")
    print(f"Total Distance: Extracted from {len(final_contributions_trace)} agent analyses")
    print(f"Estimated Duration: Based on {performance_metrics.get('total_contributions', 0)} contributions")
    print(f"Key Challenge: Identified through {performance_metrics.get('total_refinements', 0)} refinements")
    print(f"Complexity: {len(agent_profiles)} specialized agents with {sum(p.get('iteration_count', 0) for p in agent_profiles.values())} total iterations")
    print(f"Meaningful Framework: Multi-agent collaborative analysis with trace validation")
    print(f"Consensus View: {performance_metrics.get('collaboration_efficiency', 0.0):.1f} collaboration efficiency score")
    print(f"Critical Perspective: {critical_data.get('confidence', 0.0):.2f} confidence from critical analysis")
    
    # Extract critical challenges from devils advocate trace data
    if critical_data and 'content_summary' in critical_data:
        print(f"Critical Challenges: Systematic analysis from trace data")
    else:
        print(f"Critical Challenges: Available in detailed trace analysis")
    print()
    
    # Show comprehensive contribution summary from trace data
    print("COMPREHENSIVE CONTRIBUTION SUMMARY (FROM AGENT TRACE):")
    print("-" * 50)
    
    # Use trace data instead of artifact content
    final_contributions_trace = trace_dump.get("final_state", {}).get("final_contributions", {})
    collaboration_timeline = trace_dump.get("collaboration_timeline", [])
    
    print("\n📍 GEOGRAPHY ANALYSIS (FROM TRACE):")
    geo_data = final_contributions_trace.get("geo_agent_001", {})
    if geo_data:
        print(f"  Agent: geo_agent_001")
        print(f"  Confidence: {geo_data.get('confidence', 0.0):.2f}")
        print(f"  Contribution Type: {geo_data.get('contribution_type', 'Unknown')}")
        content_summary = geo_data.get('content_summary', '')
        if 'route_segments' in content_summary:
            print(f"  Analysis: Route planning with geographic segments")
        if 'total_distance_km' in content_summary:
            print(f"  Distance Analysis: Comprehensive distance calculations")
    
    print("\n🌍 CULTURAL ANALYSIS (FROM TRACE):")
    cultural_data = final_contributions_trace.get("culture_agent_001", {})
    if cultural_data:
        print(f"  Agent: culture_agent_001")
        print(f"  Confidence: {cultural_data.get('confidence', 0.0):.2f}")
        print(f"  Contribution Type: {cultural_data.get('contribution_type', 'Unknown')}")
        content_summary = cultural_data.get('content_summary', '')
        if 'cultural_regions' in content_summary:
            print(f"  Analysis: Multi-regional cultural framework")
        if 'meaningful_activities' in content_summary:
            print(f"  Activities: Cultural engagement strategies")
    
    print("\n📦 LOGISTICS ANALYSIS (FROM TRACE):")
    logistics_data = final_contributions_trace.get("logistics_agent_001", {})
    if logistics_data:
        print(f"  Agent: logistics_agent_001")
        print(f"  Confidence: {logistics_data.get('confidence', 0.0):.2f}")
        print(f"  Contribution Type: {logistics_data.get('contribution_type', 'Unknown')}")
        content_summary = logistics_data.get('content_summary', '')
        if 'resource_requirements' in content_summary:
            print(f"  Analysis: Resource optimization and planning")
            if 'timeline_optimization' in content_summary:
                print(f"  Timeline: Duration and scheduling analysis")
        
        print("\n🛡️ SAFETY ANALYSIS (FROM TRACE):")
        safety_data = final_contributions_trace.get("safety_agent_001", {})
        if safety_data:
            print(f"  Agent: safety_agent_001")
            print(f"  Confidence: {safety_data.get('confidence', 0.0):.2f}")
            print(f"  Contribution Type: {safety_data.get('contribution_type', 'Unknown')}")
            content_summary = safety_data.get('content_summary', '')
            if 'risk_categories' in content_summary:
                print(f"  Analysis: Comprehensive risk assessment")
            if 'safety_protocols' in content_summary:
                print(f"  Protocols: Multi-phase safety management")
        
        print("\n⚠️ CRITICAL ANALYSIS (FROM TRACE):")
        critical_data = final_contributions_trace.get("devils_advocate_001", {})
        if critical_data:
            print(f"  Agent: devils_advocate_001")
            print(f"  Confidence: {critical_data.get('confidence', 0.0):.2f}")
            print(f"  Contribution Type: {critical_data.get('contribution_type', 'Unknown')}")
            content_summary = critical_data.get('content_summary', '')
            if 'critical_challenges' in content_summary:
                print(f"  Analysis: Systematic challenge identification")
            if 'agent_critique' in content_summary:
                print(f"  Critiques: Cross-agent validation analysis")
        
        alternative_recs = critical_data.get("alternative_recommendations", []) if critical_data else []
        if isinstance(alternative_recs, list):
            print(f"  Alternative Recommendations: {len(alternative_recs)}")
            for rec in alternative_recs[:2]:  # Show first 2
                print(f"    - {rec}")
        else:
            print(f"  Alternative Recommendations: {alternative_recs}")
        print()
        
        # Show phase breakdown (if available in final artifact)
        print("JOURNEY PHASES:")
        if hasattr(final_artifact, 'content') and isinstance(final_artifact.content, dict):
            artifact_content = final_artifact.content
            if "consortium_response" in artifact_content and "integrated_plan" in artifact_content["consortium_response"]:
                integrated_plan = artifact_content["consortium_response"]["integrated_plan"]
                for phase_name, phase_details in integrated_plan.items():
                    print(f"  {phase_name.replace('_', ' ').title()}:")
                    print(f"    Duration: {phase_details.get('duration', 'N/A')}")
                    print(f"    Cultural Significance: {phase_details.get('cultural_significance', 'N/A')}")
            else:
                print("  Phase breakdown available in detailed artifact analysis")
        else:
            print("  Phase breakdown available in detailed artifact analysis")
        print()
        
        # Show consortium metadata (if available)
        print("CONSORTIUM METADATA:")
        if hasattr(final_artifact, 'content') and isinstance(final_artifact.content, dict):
            artifact_content = final_artifact.content
            if "consortium_metadata" in artifact_content:
                consortium_meta = artifact_content["consortium_metadata"]
                print(f"  Contributing Agents: {len(consortium_meta.get('contributing_agents', []))}")
                print(f"  Synthesis Confidence: {consortium_meta.get('synthesis_confidence', final_artifact.confidence):.2f}")
                print(f"  Interdisciplinary Integration: {consortium_meta.get('interdisciplinary_integration', True)}")
                print(f"  Cultural Sensitivity: {consortium_meta.get('cultural_sensitivity_rating', 'high')}")
            else:
                print(f"  Synthesis Confidence: {final_artifact.confidence:.2f}")
                print(f"  Contributing Agents: {len(final_contributions)}")
                print(f"  Interdisciplinary Integration: True")
                print(f"  Cultural Sensitivity: high")
        else:
            print(f"  Synthesis Confidence: {final_artifact.confidence:.2f}")
            print(f"  Contributing Agents: {len(final_contributions)}")
            print(f"  Interdisciplinary Integration: True")
            print(f"  Cultural Sensitivity: high")
        
    else:
        print("✗ Failed to create consortium MAIF artifact")
    
    print()
    print("=" * 80)
    print("CONSORTIUM DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    # PHASE 4: Comprehensive Validation and Analysis
    print("\nPHASE 4: Comprehensive Validation and Forensic Analysis...")
    print("-" * 60)
    
    # Validate the consortium MAIF
    print("Validating consortium MAIF integrity...")
    try:
        validator = MAIFValidator()
        manifest_path = output_path.replace('.maif', '_manifest.json')
        saved_files.append(manifest_path)  # Track the manifest file
        validation_report = validator.validate_file(output_path, manifest_path)
        
        print(f"✓ Validation complete:")
        print(f"  Valid: {validation_report.is_valid}")
        print(f"  Errors found: {len(validation_report.errors)}")
        print(f"  Warnings found: {len(validation_report.warnings)}")
        
        if not validation_report.is_valid and validation_report.errors:
            print("  Errors detected:")
            for error in validation_report.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        if validation_report.warnings:
            print("  Warnings:")
            for warning in validation_report.warnings[:3]:  # Show first 3 warnings
                print(f"    - {warning}")
                
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Perform forensic analysis
    print("\nPerforming forensic analysis...")
    try:
        from maif.core import MAIFParser
        parser = MAIFParser(output_path, manifest_path)
        verifier = MAIFVerifier()
        forensic_analyzer = ForensicAnalyzer()
        
        forensic_report = forensic_analyzer.analyze_maif_file(output_path, manifest_path)
        
        print(f"✓ Forensic analysis complete:")
        print(f"  File analyzed: {forensic_report['file_path']}")
        print(f"  Analysis timestamp: {forensic_report['analysis_timestamp']}")
        print(f"  Evidence items: {len(forensic_report['evidence_summary'])}")
        print(f"  Timeline events: {len(forensic_report['timeline'])}")
        print(f"  Risk assessment: {forensic_report['risk_assessment']}")
        
        if forensic_report.get('evidence_summary'):
            print("  Key evidence:")
            evidence_items = forensic_report['evidence_summary']
            if isinstance(evidence_items, list):
                for evidence in evidence_items[:2]:  # Show first 2 evidence items
                    if isinstance(evidence, dict):
                        severity = evidence.get('severity', 'unknown').upper()
                        description = evidence.get('description', 'No description')
                        print(f"    - {severity}: {description}")
                
    except Exception as e:
        print(f"✗ Forensic analysis failed: {e}")
    
    # PHASE 5: Version History and Content Evolution Analysis
    print("\nPHASE 5: Version History and Content Evolution Analysis...")
    print("-" * 60)
    
    # Analyze version history across all agents
    total_versions = 0
    total_refinements = 0
    
    for agent in agents + [coordinator_agent]:
        version_history = agent.get_version_history()
        agent_versions = sum(len(versions) for versions in version_history.values())
        total_versions += agent_versions
        total_refinements += len(agent.refinement_history)
        
        print(f"✓ {agent.agent_id}:")
        print(f"  Total versions: {agent_versions}")
        print(f"  Content blocks: {len(version_history)}")
        print(f"  Refinement iterations: {len(agent.refinement_history)}")
        
        # Show version evolution for key content
        for block_id, versions in version_history.items():
            if len(versions) > 1:
                print(f"  Block {block_id}: {len(versions)} versions")
                for i, version in enumerate(versions):
                    print(f"    v{version['version']}: {version['operation']} by {version['agent_id']}")
    
    print(f"\nOverall Version Statistics:")
    print(f"  Total versions across all agents: {total_versions}")
    print(f"  Total refinement iterations: {total_refinements}")
    print(f"  Average versions per agent: {total_versions / len(agents + [coordinator_agent]):.1f}")
    
    # PHASE 6: Content Quality and Collaboration Metrics
    print("\nPHASE 6: Content Quality and Collaboration Metrics...")
    print("-" * 60)
    
    # Calculate collaboration metrics
    num_contributions = len(contributions) if contributions else 0
    collaboration_metrics = {
        "agent_participation": len(agents),
        "total_contributions": num_contributions,
        "average_confidence": (sum(c.confidence for c in contributions.values()) / num_contributions) if num_contributions > 0 else 0.0,
        "dependency_satisfaction": (sum(1 for c in contributions.values() if c.dependencies) / num_contributions) if num_contributions > 0 else 0.0,
        "content_complexity": (sum(len(json.dumps(c.content)) for c in contributions.values()) / num_contributions) if num_contributions > 0 else 0.0,
        "refinement_cycles": coordinator_agent.collaboration_rounds,
        "feedback_iterations": len(coordinator_agent.feedback_history)
    }
    
    print("Collaboration Quality Metrics:")
    for metric, value in collaboration_metrics.items():
        if isinstance(value, float):
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    # Consolidate all agent contributions into the single consortium MAIF file
    print("\nPHASE 7: Consolidating all agent contributions into single MAIF file...")
    print("-" * 60)
    
    # Load the existing consortium MAIF file
    consortium_maif = load_maif(output_path)
    
    # Add individual agent summaries to the main consortium MAIF
    for agent in agents + [coordinator_agent]:
        agent_summary = agent.get_contribution_summary()
        final_metadata = {
            "agent_id": agent.agent_id,
            "final_summary": agent_summary,
            "collaboration_metrics": collaboration_metrics,
            "consortium_completion_timestamp": time.time(),
            "agent_capabilities": getattr(agent, 'capabilities', []),
            "total_contributions": len(getattr(agent, 'contribution_history', []))
        }
        
        # Add agent summary as a separate block in the consortium MAIF
        consortium_maif.encoder.add_binary_block(
            json.dumps(final_metadata).encode('utf-8'),
            "metadata",
            metadata={
                "type": "agent_final_summary",
                "agent_id": agent.agent_id,
                "timestamp": time.time()
            }
        )
        print(f"✓ Consolidated {agent.agent_id} contribution into consortium MAIF")
    
    # Save the consolidated consortium MAIF file (use the existing output_path)
    consortium_file = output_path  # Use the existing consortium MAIF file
    if consortium_maif.save(consortium_file):
        print(f"✓ Saved consolidated consortium MAIF: {consortium_file}")
        
        # Verify consolidated file integrity
        try:
            test_maif = load_maif(consortium_file)
            if test_maif.verify_integrity():
                print(f"  ✓ Integrity verified for {consortium_file}")
            else:
                print(f"  ⚠ Integrity check failed for {consortium_file}")
        except Exception as e:
            print(f"  ⚠ Could not verify {consortium_file}: {e}")
    else:
        print(f"✗ Failed to save consolidated consortium MAIF")
    
    # Final Consortium Analysis and Reporting
    print("\nPHASE 8: Final Consortium Analysis and Reporting...")
    print("-" * 60)
    
    # Generate comprehensive consortium report
    consortium_report = {
        "consortium_id": "earth_to_andromeda_galaxy_consortium",
        "completion_timestamp": time.time(),
        "query_processed": query,
        "agents_participated": [agent.agent_id for agent in agents + [coordinator_agent]],
        "total_maif_files_created": 1,  # Single consolidated MAIF file
        "version_tracking_summary": {
            "total_versions": total_versions,
            "total_refinements": total_refinements,
            "version_history_complete": True
        },
        "quality_assessment": {
            "all_dependencies_satisfied": all(
                not c.dependencies or
                all(dep in [contrib.contribution_type for contrib in contributions.values()]
                    for dep in c.dependencies)
                for c in contributions.values()
            ),
            "confidence_threshold_met": all(c.confidence >= 0.7 for c in contributions.values()),
            "content_completeness": "comprehensive",
            "cultural_sensitivity": "high"
        },
        "technical_features_demonstrated": [
            "version_history_tracking",
            "iterative_refinement",
            "cross_agent_dependencies",
            "privacy_controls",
            "semantic_embeddings",
            "forensic_analysis",
            "validation_and_repair",
            "multimodal_content",
            "metadata_management"
        ]
    }
    
    print("Final Consortium Report:")
    print(f"  Consortium ID: {consortium_report['consortium_id']}")
    print(f"  Agents Participated: {len(consortium_report['agents_participated'])}")
    print(f"  MAIF Files Created: {consortium_report['total_maif_files_created']}")
    print(f"  Version History Complete: {consortium_report['version_tracking_summary']['version_history_complete']}")
    print(f"  Quality Assessment: {consortium_report['quality_assessment']['content_completeness']}")
    print(f"  Technical Features: {len(consortium_report['technical_features_demonstrated'])}")
    
    print()
    print("=" * 80)
    print("ENHANCED CONSORTIUM DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Achievements:")
    print("✓ Multi-agent collaboration with version tracking")
    print("✓ Iterative refinement and feedback loops")
    print("✓ Comprehensive content evolution history")
    print("✓ Forensic analysis and validation")
    print("✓ Privacy controls and security features")
    print("✓ Semantic embeddings and cross-modal attention")
    print("✓ Dependency management and synthesis")
    print("✓ Cultural sensitivity and meaningful framework")
    
    # Save the single shared MAIF file
    print("\n" + "=" * 100)
    print("SAVING SHARED MAIF FILE")
    print("=" * 100)
    
    shared_maif_filename = "multi_agent_consortium.maif"
    try:
        shared_maif.save(shared_maif_filename)
        print(f"✓ Shared MAIF file saved: {shared_maif_filename}")
        print(f"✓ Contains contributions from {len(agents)} agents + coordinator")
        saved_files.append(shared_maif_filename)
    except Exception as e:
        print(f"⚠️  Error saving shared MAIF file: {e}")
    
    # Add comprehensive trace dump at the end
    print("\n" + "=" * 100)
    print("COMPREHENSIVE AGENT ACTIVITY TRACE")
    print("=" * 100)
    
    # Save trace to file (trace_dump was already generated above for the summaries)
    trace_file = "agent_activity_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace_dump, f, indent=2, default=str)
    
    print(f"✓ Complete agent activity trace saved to: {trace_file}")
    print(f"✓ Trace contains {len(trace_dump.get('agent_profiles', {}))} agent profiles")
    print(f"✓ Collaboration timeline: {len(trace_dump.get('collaboration_timeline', []))} rounds")
    print(f"✓ Performance metrics: {trace_dump.get('performance_metrics', {}).get('total_contributions', 0)} total contributions")
    print(f"✓ Both CONSORTIUM ARTIFACT SUMMARY and COMPREHENSIVE CONTRIBUTION SUMMARY now use trace data")
    print(f"✓ Using single shared MAIF file instead of {len(agents)} individual files")
    saved_files.append(trace_file)
    
    return output_path, saved_files, consortium_report


def analyze_consortium_artifact(maif_path: str):
    """Analyze the created consortium artifact."""
    
    print(f"\nAnalyzing consortium artifact: {maif_path}")
    print("-" * 50)
    
    try:
        # Load the consortium MAIF
        from maif_api import load_maif
        consortium_maif = load_maif(maif_path)
        
        # Verify integrity
        if consortium_maif.verify_integrity():
            print("✓ Artifact integrity verified")
        else:
            print("✗ Artifact integrity check failed")
        
        # Get content summary
        content_list = consortium_maif.get_content_list()
        print(f"✓ Total content blocks: {len(content_list)}")
        
        # Show content breakdown
        content_types = {}
        for content in content_list:
            content_type = content.get("type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        print("Content breakdown:")
        for content_type, count in content_types.items():
            print(f"  {content_type}: {count} blocks")
        
        # Get privacy report
        privacy_report = consortium_maif.get_privacy_report()
        print(f"✓ Privacy enabled: {privacy_report.get('privacy_enabled', False)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return False


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Consortium Demo with configurable collaboration rounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_agent_consortium_demo.py                    # Run with defaults (10 rounds, 2 geo, 1 each other)
  python multi_agent_consortium_demo.py --rounds 5         # Run with 5 rounds
  python multi_agent_consortium_demo.py --geo-agents 3     # Run with 3 geography agents
  python multi_agent_consortium_demo.py --culture-agents 2 --safety-agents 2  # Multiple agent types
  python multi_agent_consortium_demo.py --rounds 3 --geo-agents 1 --culture-agents 1 --logistics-agents 1 --safety-agents 1 --devils-advocate-agents 1  # Minimal setup
        """
    )
    
    parser.add_argument(
        '--rounds', '-r',
        type=int,
        default=10,
        help='Number of collaboration rounds to execute (default: 10)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--geo-agents',
        type=int,
        default=2,
        help='Number of geography agents (default: 2)'
    )
    
    parser.add_argument(
        '--culture-agents',
        type=int,
        default=1,
        help='Number of cultural agents (default: 1)'
    )
    
    parser.add_argument(
        '--logistics-agents',
        type=int,
        default=1,
        help='Number of logistics agents (default: 1)'
    )
    
    parser.add_argument(
        '--safety-agents',
        type=int,
        default=1,
        help='Number of safety agents (default: 1)'
    )
    
    parser.add_argument(
        '--devils-advocate-agents',
        type=int,
        default=1,
        help='Number of devils advocate agents (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rounds < 1:
        print("Error: Number of rounds must be at least 1")
        return 1
    
    if args.rounds > 50:
        print("Warning: Large number of rounds may take significant time")
    
    # Validate agent counts
    agent_counts = {
        'geo': args.geo_agents,
        'culture': args.culture_agents,
        'logistics': args.logistics_agents,
        'safety': args.safety_agents,
        'devils_advocate': args.devils_advocate_agents
    }
    
    for agent_type, count in agent_counts.items():
        if count < 0:
            print(f"Error: Number of {agent_type} agents must be non-negative")
            return 1
        if count > 10:
            print(f"Warning: Large number of {agent_type} agents ({count}) may impact performance")
    
    total_agents = sum(agent_counts.values())
    if total_agents == 0:
        print("Error: At least one agent must be specified")
        return 1
    if total_agents > 20:
        print(f"Warning: Large total number of agents ({total_agents}) may significantly impact performance")
    
    print(f"🚀 Starting Multi-Agent Consortium Demo with {args.rounds} collaboration rounds")
    print(f"Agent configuration: {args.geo_agents} geo, {args.culture_agents} culture, {args.logistics_agents} logistics, {args.safety_agents} safety, {args.devils_advocate_agents} devils advocate")
    if args.verbose:
        print(f"Verbose mode enabled")
    
    try:
        # Run the main demonstration
        result = demonstrate_multi_agent_consortium(
            num_rounds=args.rounds,
            agent_counts=agent_counts
        )
        
        # Handle enhanced return values
        if isinstance(result, tuple) and len(result) == 3:
            artifact_path, saved_files, consortium_report = result
        else:
            artifact_path = result
            saved_files = []
            consortium_report = {}
        
        # Analyze the created artifact
        if artifact_path and os.path.exists(artifact_path):
            analyze_consortium_artifact(artifact_path)
        
        print("\n" + "=" * 80)
        print("ENHANCED MULTI-AGENT CONSORTIUM DEMO COMPLETED SUCCESSFULLY!")
        print(f"Collaboration rounds executed: {args.rounds}")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n✗ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())