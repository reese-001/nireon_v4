# tests/domain/mixins/test_deterministic_rng_mixin.py

"""
run this test with:
# Test the mixin itself
python -m pytest tests/domain/mixins/test_deterministic_rng_mixin.py -v

# Test Explorer integration
python -m pytest tests/components/mechanisms/explorer/test_explorer_service.py::test_frame_rng_deterministic_behavior -v

# Run all Explorer tests to make sure nothing broke
python -m pytest tests/components/mechanisms/explorer/test_explorer_service.py -v
"""



from unittest.mock import AsyncMock
import pytest
import time
import random
from domain.frames import Frame
from domain.mixins.deterministic_rng_mixin import DeterministicRNGMixin

class TestDeterministicRNGMixin:
    """Test suite for DeterministicRNGMixin functionality."""

    def test_frame_rng_is_deterministic(self):
        """Test that frames with identical properties produce identical RNG sequences."""
        # Create two frames with identical properties
        frame_id = "F-123"
        created_ts = 1718000000.0
        
        f1 = Frame(
            id=frame_id,
            name="Test Frame 1", 
            description="Test",
            owner_agent_id="test_agent",
            created_ts=created_ts,
            updated_ts=created_ts
        )
        
        f2 = Frame(
            id=frame_id,
            name="Test Frame 2",  # Different name shouldn't affect RNG
            description="Test",
            owner_agent_id="test_agent", 
            created_ts=created_ts,
            updated_ts=created_ts
        )
        
        # Get RNG instances
        rng1 = f1.get_rng()
        rng2 = f2.get_rng()
        
        # Generate sequences and compare
        sequence1 = [rng1.random() for _ in range(10)]
        sequence2 = [rng2.random() for _ in range(10)]
        
        assert sequence1 == sequence2, "Identical frames should produce identical RNG sequences"
        
        # Test integer generation
        rng1_reset = f1.get_rng()  # Should return cached instance
        rng2_reset = f2.get_rng()  # Should return cached instance
        
        # Reset both to start of sequence again
        f1.reset_rng()
        f2.reset_rng()
        
        rng1_fresh = f1.get_rng()
        rng2_fresh = f2.get_rng()
        
        int_sequence1 = [rng1_fresh.randint(0, 100) for _ in range(5)]
        int_sequence2 = [rng2_fresh.randint(0, 100) for _ in range(5)]
        
        assert int_sequence1 == int_sequence2, "Integer sequences should also be identical"

    def test_different_frames_produce_different_sequences(self):
        """Test that frames with different properties produce different RNG sequences."""
        base_time = 1718000000.0
        
        f1 = Frame(
            id="F-123",
            name="Test Frame",
            description="Test", 
            owner_agent_id="test_agent",
            created_ts=base_time,
            updated_ts=base_time
        )
        
        f2 = Frame(
            id="F-456",  # Different ID
            name="Test Frame",
            description="Test",
            owner_agent_id="test_agent", 
            created_ts=base_time,
            updated_ts=base_time
        )
        
        f3 = Frame(
            id="F-123",
            name="Test Frame", 
            description="Test",
            owner_agent_id="test_agent",
            created_ts=base_time + 1.0,  # Different creation time
            updated_ts=base_time + 1.0
        )
        
        # Generate sequences
        sequence1 = [f1.get_rng().random() for _ in range(10)]
        sequence2 = [f2.get_rng().random() for _ in range(10)]
        sequence3 = [f3.get_rng().random() for _ in range(10)]
        
        # All should be different
        assert sequence1 != sequence2, "Different frame IDs should produce different sequences"
        assert sequence1 != sequence3, "Different creation times should produce different sequences"
        assert sequence2 != sequence3, "All different frames should produce different sequences"

    def test_rng_caching(self):
        """Test that get_rng() returns the same instance when called multiple times."""
        frame = Frame(
            id="F-cache-test",
            name="Cache Test",
            description="Test",
            owner_agent_id="test_agent",
            created_ts=time.time(),
            updated_ts=time.time()
        )
        
        rng1 = frame.get_rng()
        rng2 = frame.get_rng()
        
        # Should be the exact same object
        assert rng1 is rng2, "get_rng() should return cached instance"
        
        # Should continue from same state
        val1 = rng1.random()
        val2 = rng2.random()
        
        # Reset and verify they share state
        frame.reset_rng()
        rng3 = frame.get_rng()
        
        # Should be a new instance after reset
        assert rng3 is not rng1, "reset_rng() should create new instance on next get_rng()"

    def test_custom_seed_override(self):
        """Test custom seed functionality."""
        frame = Frame(
            id="F-custom-seed",
            name="Custom Seed Test",
            description="Test",
            owner_agent_id="test_agent", 
            created_ts=time.time(),
            updated_ts=time.time()
        )
        
        # Get deterministic sequence
        original_sequence = [frame.get_rng().random() for _ in range(5)]
        
        # Set custom seed
        custom_seed = 42
        frame.set_custom_rng_seed(custom_seed)
        
        custom_sequence = [frame.get_rng().random() for _ in range(5)]
        
        # Should be different from original
        assert custom_sequence != original_sequence, "Custom seed should produce different sequence"
        
        # Verify custom seed produces consistent results
        frame.set_custom_rng_seed(custom_seed)  # Set same seed again
        custom_sequence2 = [frame.get_rng().random() for _ in range(5)]
        
        assert custom_sequence == custom_sequence2, "Same custom seed should produce same sequence"
        
        # Revert to deterministic
        frame.set_custom_rng_seed(None)
        reverted_sequence = [frame.get_rng().random() for _ in range(5)]
        
        assert reverted_sequence == original_sequence, "Reverting should restore deterministic sequence"

    def test_seed_generation_consistency(self):
        """Test that seed generation is consistent."""
        frame = Frame(
            id="F-seed-test",
            name="Seed Test",
            description="Test",
            owner_agent_id="test_agent",
            created_ts=1718000000.0,
            updated_ts=1718000000.0
        )
        
        # Get seed multiple times
        seed1 = frame.get_rng_seed()
        seed2 = frame.get_rng_seed()
        
        assert seed1 == seed2, "get_rng_seed() should be consistent"
        
        # Verify RNG uses this seed
        frame.reset_rng()
        rng = frame.get_rng()
        
        # Create new RNG with same seed manually
        manual_rng = random.Random(seed1)
        
        # Should produce same sequences
        rng_sequence = [rng.random() for _ in range(10)]
        manual_sequence = [manual_rng.random() for _ in range(10)]
        
        assert rng_sequence == manual_sequence, "RNG should use the generated seed"

    def test_schema_version_affects_seed(self):
        """Test that schema version affects seed generation."""
        base_time = 1718000000.0
        
        f1 = Frame(
            id="F-schema-test",
            name="Schema Test",
            description="Test",
            owner_agent_id="test_agent",
            created_ts=base_time,
            updated_ts=base_time,
            schema_version="1.0"
        )
        
        f2 = Frame(
            id="F-schema-test", 
            name="Schema Test",
            description="Test",
            owner_agent_id="test_agent",
            created_ts=base_time,
            updated_ts=base_time,
            schema_version="2.0"  # Different schema version
        )
        
        seed1 = f1.get_rng_seed()
        seed2 = f2.get_rng_seed()
        
        assert seed1 != seed2, "Different schema versions should produce different seeds"

    def test_mixin_integration_with_explorer(self):
        """Test integration with ExplorerMechanism-style usage."""
        frame = Frame(
            id="F-explorer-integration",
            name="Explorer Integration Test", 
            description="Test frame for explorer integration",
            owner_agent_id="explorer_v4",
            created_ts=time.time(),
            updated_ts=time.time()
        )
        
        # Simulate explorer-style usage
        frame_rng = frame.get_rng()
        
        # Test various random operations explorer might do
        random_choices = []
        for _ in range(5):
            # Simulate choosing from a list of strategies
            strategies = ["depth_first", "breadth_first", "random", "guided"]
            choice = frame_rng.choice(strategies)
            random_choices.append(choice)
        
        # Test random floats for creativity factors
        creativity_factors = [frame_rng.random() for _ in range(3)]
        
        # Test random integers for variation counts
        variation_counts = [frame_rng.randint(1, 10) for _ in range(3)]
        
        # Reset and verify reproducibility
        frame.reset_rng()
        frame_rng_reset = frame.get_rng()
        
        random_choices_2 = []
        for _ in range(5):
            strategies = ["depth_first", "breadth_first", "random", "guided"]
            choice = frame_rng_reset.choice(strategies)
            random_choices_2.append(choice)
        
        creativity_factors_2 = [frame_rng_reset.random() for _ in range(3)]
        variation_counts_2 = [frame_rng_reset.randint(1, 10) for _ in range(3)]
        
        # All should be identical
        assert random_choices == random_choices_2
        assert creativity_factors == creativity_factors_2  
        assert variation_counts == variation_counts_2

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Very long frame ID
        long_id = "F-" + "x" * 1000
        frame_long_id = Frame(
            id=long_id,
            name="Long ID Test",
            description="Test",
            owner_agent_id="test_agent",
            created_ts=time.time(),
            updated_ts=time.time()
        )
        
        # Should handle long IDs gracefully
        rng = frame_long_id.get_rng()
        assert isinstance(rng, random.Random)
        
        # Special characters in ID
        special_id = "F-test!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        frame_special = Frame(
            id=special_id,
            name="Special ID Test", 
            description="Test",
            owner_agent_id="test_agent",
            created_ts=time.time(),
            updated_ts=time.time()
        )
        
        rng_special = frame_special.get_rng()
        assert isinstance(rng_special, random.Random)
        
        # Zero timestamp
        frame_zero_time = Frame(
            id="F-zero-time",
            name="Zero Time Test",
            description="Test", 
            owner_agent_id="test_agent",
            created_ts=0.0,
            updated_ts=0.0
        )
        
        rng_zero = frame_zero_time.get_rng()
        assert isinstance(rng_zero, random.Random)

# Additional test to add to your existing explorer tests
class TestExplorerWithDeterministicRNG:
    """Tests for ExplorerMechanism using deterministic RNG."""
    
    @pytest.mark.asyncio
    async def test_explorer_deterministic_behavior(self, explorer_mechanism, mock_nireon_execution_context, mock_gateway_port, mock_frame_factory_service):
        """Test that explorer produces deterministic results with frame RNG."""
        await explorer_mechanism.initialize(mock_nireon_execution_context)
        
        # Override frame creation to use deterministic frames
        deterministic_frame_id = "deterministic_test_frame"
        deterministic_created_ts = 1718000000.0
        
        async def create_deterministic_frame(*args, **kwargs):
            from domain.frames import Frame
            frame = Frame(
                id=deterministic_frame_id,
                name=kwargs.get("name", "test"),
                description=kwargs.get("description", "test"),
                owner_agent_id=kwargs.get("owner_agent_id", "test"),
                created_ts=deterministic_created_ts,
                updated_ts=deterministic_created_ts,
                status=kwargs.get("initial_status", "active"),
                epistemic_goals=kwargs.get("epistemic_goals", []),
                llm_policy=kwargs.get("llm_policy", {}),
                context_tags=kwargs.get("context_tags", {"audit_trail": []})
            )
            return frame
        
        mock_frame_factory_service.create_frame.side_effect = create_deterministic_frame
        
        # Mock deterministic LLM responses that use frame RNG
        async def deterministic_llm_response(ce):
            if ce.service_call_type == 'LLM_ASK':
                # Use a deterministic response that could vary based on RNG
                return {"text": f"Deterministic response for {ce.frame_id}", "usage": {"total_tokens": 50}}
            return {}
        
        mock_gateway_port.process_cognitive_event = AsyncMock(side_effect=deterministic_llm_response)
        
        # Run explorer twice with same input
        seed_data = "A deterministic test case"
        
        result1 = await explorer_mechanism.process(seed_data, mock_nireon_execution_context)
        
        # Reset mocks for second run  
        mock_gateway_port.reset_mock()
        mock_frame_factory_service.reset_mock()
        mock_frame_factory_service.create_frame.side_effect = create_deterministic_frame
        mock_gateway_port.process_cognitive_event = AsyncMock(side_effect=deterministic_llm_response)
        
        result2 = await explorer_mechanism.process(seed_data, mock_nireon_execution_context)
        
        # Results should be identical (same number of variations, same frame behavior)
        assert result1.output_data["variations_generated_count"] == result2.output_data["variations_generated_count"]
        assert result1.success == result2.success