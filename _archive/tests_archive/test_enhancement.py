"""
Test Enhancement System
Simplified version for testing iterative story improvement
"""

import asyncio
import aiohttp
import time
from typing import Dict
from dataclasses import dataclass
from enum import Enum

class SceneQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    ADEQUATE = 3
    WEAK = 2
    POOR = 1

@dataclass
class TestScene:
    id: int
    name: str
    situation: str
    tension: float
    quality: SceneQuality = SceneQuality.ADEQUATE
    
class QuickEnhancementTest:
    """Quick test of enhancement system"""
    
    def __init__(self, model: str = "google/gemma-2-27b"):
        self.model = model
        self.url = "http://localhost:1234/v1/chat/completions"
        self.enhancement_history = []
        
    async def call_llm(self, prompt: str, max_tokens: int = 400) -> str:
        """Simple LLM call"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8,
            "max_tokens": max_tokens
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload) as response:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM error: {e}")
            return ""
    
    def evaluate_scene_quality(self, scene: TestScene) -> Dict:
        """Simple quality evaluation"""
        
        # Basic heuristics
        quality_score = 3  # Start at adequate
        
        # Check length
        if len(scene.situation) > 200:
            quality_score += 1
        if len(scene.situation) > 300:
            quality_score += 1
            
        # Check for dramatic elements
        dramatic_words = ['suddenly', 'tension', 'conflict', 'desperate', 'critical', 
                         'dangerous', 'revealed', 'shocked', 'betrayed', 'discovered']
        
        situation_lower = scene.situation.lower()
        drama_count = sum(1 for word in dramatic_words if word in situation_lower)
        
        if drama_count >= 3:
            quality_score = min(5, quality_score + 1)
        
        # Adjust for tension
        if scene.tension > 0.7:
            quality_score = min(5, quality_score + 1)
        
        quality = SceneQuality(min(5, max(1, quality_score)))
        
        weaknesses = []
        if quality_score < 4:
            weaknesses.append("Needs more dramatic tension")
        if len(scene.situation) < 150:
            weaknesses.append("Too brief, needs expansion")
        if drama_count < 2:
            weaknesses.append("Lacks conflict or stakes")
            
        return {
            "quality": quality,
            "score": quality_score,
            "weaknesses": weaknesses,
            "drama_elements": drama_count
        }
    
    async def enhance_scene(self, scene: TestScene, focus: str = "conflict") -> TestScene:
        """Enhance a scene"""
        
        enhancement_prompts = {
            "conflict": "Add tension, opposition, and dramatic conflict",
            "emotion": "Deepen emotional stakes and character feelings",
            "action": "Add dynamic physical actions and movement",
            "mystery": "Introduce questions, secrets, or uncertainty"
        }
        
        prompt = f"""Enhance this scene by adding {enhancement_prompts.get(focus, 'more drama')}:

Original scene: {scene.situation}

Create an enhanced version that is more dramatic and engaging. Include specific details and heightened stakes.

Enhanced scene:"""
        
        enhanced_text = await self.call_llm(prompt)
        
        if enhanced_text:
            enhanced_scene = TestScene(
                id=scene.id,
                name=scene.name + f" [Enhanced-{focus}]",
                situation=enhanced_text,
                tension=min(1.0, scene.tension + 0.1)
            )
            return enhanced_scene
        
        return scene
    
    async def run_enhancement_test(self):
        """Run a complete enhancement test"""
        
        print("🧪 ENHANCEMENT SYSTEM TEST")
        print("=" * 60)
        
        # Create test scenes
        test_scenes = [
            TestScene(
                id=1,
                name="Opening",
                situation="The detective arrives at the crime scene. It's raining.",
                tension=0.3
            ),
            TestScene(
                id=2,
                name="Discovery",
                situation="A clue is found that changes everything.",
                tension=0.5
            ),
            TestScene(
                id=3,
                name="Confrontation",
                situation="The detective confronts the suspect.",
                tension=0.8
            )
        ]
        
        print(f"\n📝 Testing with {len(test_scenes)} scenes\n")
        
        for scene in test_scenes:
            print(f"{'='*60}")
            print(f"🎬 SCENE {scene.id}: {scene.name}")
            print(f"📊 Initial tension: {scene.tension:.0%}")
            
            # Original scene
            print(f"\n📄 Original ({len(scene.situation)} chars):")
            print(f"   {scene.situation}")
            
            # Evaluate original
            eval_original = self.evaluate_scene_quality(scene)
            print(f"\n📊 Original Quality: {eval_original['quality'].name} ({eval_original['score']}/5)")
            if eval_original['weaknesses']:
                print(f"   Weaknesses: {', '.join(eval_original['weaknesses'])}")
            
            # Enhance
            print("\n🔧 Enhancing scene...")
            start = time.time()
            
            # Try different enhancement focuses
            focus = "conflict" if scene.tension < 0.6 else "emotion"
            enhanced_scene = await self.enhance_scene(scene, focus)
            
            enhancement_time = time.time() - start
            print(f"⏱️  Enhanced in {enhancement_time:.1f}s")
            
            # Show enhanced version
            if enhanced_scene.situation != scene.situation:
                print(f"\n📄 Enhanced ({len(enhanced_scene.situation)} chars):")
                print(f"   {enhanced_scene.situation[:300]}{'...' if len(enhanced_scene.situation) > 300 else ''}")
                
                # Evaluate enhanced
                eval_enhanced = self.evaluate_scene_quality(enhanced_scene)
                print(f"\n📊 Enhanced Quality: {eval_enhanced['quality'].name} ({eval_enhanced['score']}/5)")
                
                # Calculate improvement
                improvement = eval_enhanced['score'] - eval_original['score']
                if improvement > 0:
                    print(f"✅ Improvement: +{improvement} points")
                elif improvement == 0:
                    print("➖ No change in quality score")
                else:
                    print(f"⚠️  Quality decreased: {improvement} points")
                
                # Store history
                self.enhancement_history.append({
                    'scene': scene.name,
                    'original_score': eval_original['score'],
                    'enhanced_score': eval_enhanced['score'],
                    'improvement': improvement,
                    'focus': focus,
                    'time': enhancement_time
                })
            else:
                print("⚠️  Enhancement failed")
        
        # Summary
        if self.enhancement_history:
            print(f"\n{'='*60}")
            print("📊 ENHANCEMENT SUMMARY")
            print(f"{'='*60}")
            
            total_improvement = sum(h['improvement'] for h in self.enhancement_history)
            avg_time = sum(h['time'] for h in self.enhancement_history) / len(self.enhancement_history)
            
            print(f"\nScenes enhanced: {len(self.enhancement_history)}")
            print(f"Total improvement: +{total_improvement} points")
            print(f"Average time per scene: {avg_time:.1f}s")
            
            print("\nPer-scene results:")
            for h in self.enhancement_history:
                symbol = "✅" if h['improvement'] > 0 else "➖"
                print(f"  {symbol} {h['scene']}: {h['original_score']} → {h['enhanced_score']} (focus: {h['focus']})")

async def test_iterative_enhancement():
    """Test iterative enhancement with multiple passes"""
    
    print("\n" + "="*60)
    print("🔄 ITERATIVE ENHANCEMENT TEST")
    print("="*60)
    
    # Create a weak scene
    scene = TestScene(
        id=1,
        name="Critical Moment",
        situation="Something important happens in the story.",
        tension=0.4
    )
    
    print(f"\n📝 Starting scene: {scene.situation}")
    
    tester = QuickEnhancementTest()
    
    # Run 3 enhancement iterations
    current_scene = scene
    iteration_results = []
    
    for i in range(3):
        print(f"\n🔧 Enhancement Pass {i+1}")
        print("-" * 40)
        
        # Evaluate current
        eval = tester.evaluate_scene_quality(current_scene)
        print(f"Current quality: {eval['quality'].name} ({eval['score']}/5)")
        
        if eval['score'] >= 4:
            print("✅ Quality threshold reached!")
            break
        
        # Select focus based on weaknesses
        if "conflict" in str(eval['weaknesses']).lower():
            focus = "conflict"
        elif "emotion" in str(eval['weaknesses']).lower():
            focus = "emotion"
        else:
            focus = "mystery"
        
        print(f"Focus: {focus}")
        
        # Enhance
        enhanced = await tester.enhance_scene(current_scene, focus)
        
        if enhanced.situation != current_scene.situation:
            print(f"Enhanced: {enhanced.situation[:150]}...")
            
            # Evaluate improvement
            new_eval = tester.evaluate_scene_quality(enhanced)
            improvement = new_eval['score'] - eval['score']
            
            iteration_results.append({
                'iteration': i+1,
                'before': eval['score'],
                'after': new_eval['score'],
                'improvement': improvement
            })
            
            print(f"Result: {eval['score']} → {new_eval['score']} ({'+' if improvement >= 0 else ''}{improvement})")
            
            current_scene = enhanced
        else:
            print("Enhancement failed")
            break
    
    # Show final result
    print("\n📊 FINAL RESULT")
    print("-" * 40)
    print(f"Original: {scene.situation}")
    print(f"\nFinal: {current_scene.situation[:300]}...")
    
    if iteration_results:
        total_improvement = sum(r['improvement'] for r in iteration_results)
        print(f"\nTotal improvement: +{total_improvement} points over {len(iteration_results)} iterations")

async def test_parallel_enhancement():
    """Test enhancing the same scene with different focuses"""
    
    print("\n" + "="*60)
    print("🔀 PARALLEL ENHANCEMENT TEST")
    print("="*60)
    
    scene = TestScene(
        id=1,
        name="Pivotal Scene",
        situation="The hero must make a choice that will affect everyone.",
        tension=0.6
    )
    
    print(f"\n📝 Base scene: {scene.situation}")
    
    tester = QuickEnhancementTest()
    
    # Enhance with different focuses
    focuses = ["conflict", "emotion", "action", "mystery"]
    variants = []
    
    print("\n🔧 Creating variants with different enhancement focuses...")
    
    for focus in focuses:
        print(f"\n  {focus.upper()} variant:")
        enhanced = await tester.enhance_scene(scene, focus)
        
        if enhanced.situation != scene.situation:
            eval = tester.evaluate_scene_quality(enhanced)
            variants.append({
                'focus': focus,
                'scene': enhanced,
                'quality': eval['quality'],
                'score': eval['score'],
                'text': enhanced.situation
            })
            print(f"    Quality: {eval['quality'].name} ({eval['score']}/5)")
            print(f"    Preview: {enhanced.situation[:100]}...")
    
    # Select best variant
    if variants:
        best = max(variants, key=lambda v: v['score'])
        
        print(f"\n🏆 BEST VARIANT: {best['focus'].upper()}")
        print(f"Quality: {best['quality'].name} ({best['score']}/5)")
        print(f"\nFull text: {best['text'][:400]}...")

async def main():
    """Run all tests"""
    
    print("\n🚀 TESTING ENHANCEMENT SYSTEMS\n")
    
    # Test 1: Basic enhancement
    test = QuickEnhancementTest()
    await test.run_enhancement_test()
    
    # Test 2: Iterative enhancement
    await test_iterative_enhancement()
    
    # Test 3: Parallel variants
    await test_parallel_enhancement()
    
    print("\n✨ All tests complete!")

if __name__ == "__main__":
    asyncio.run(main())