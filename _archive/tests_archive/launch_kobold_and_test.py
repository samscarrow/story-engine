"""
Launch KoboldCpp server and run feedback collection
Automates the process of starting the server and running tests
"""

import subprocess
import asyncio
import os


async def check_server_ready(max_attempts=30):
    """Check if KoboldCpp server is ready"""
    import aiohttp

    url = "http://localhost:5001/api/v1/model"

    for i in range(max_attempts):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=2) as response:
                    if response.status == 200:
                        print("✅ KoboldCpp server is ready!")
                        return True
        except:
            pass

        print(f"⏳ Waiting for server... ({i+1}/{max_attempts})")
        await asyncio.sleep(2)

    return False


async def run_feedback_collection():
    """Run the feedback collection script"""
    from get_kobold_feedback_balanced import main as get_feedback

    await get_feedback()


async def main():
    """Launch KoboldCpp and run tests"""

    print("🚀 KOBOLDCPP TEST LAUNCHER")
    print("=" * 70)

    # KoboldCpp configuration
    kobold_exe = "./koboldcpp.exe"
    model_path = "/c/Users/sscar/.lmstudio/models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"

    # Check if executable exists
    if not os.path.exists(kobold_exe):
        print(f"❌ KoboldCpp executable not found at: {kobold_exe}")
        print("Please ensure koboldcpp.exe is in the current directory")
        return

    # Launch KoboldCpp
    cmd = [kobold_exe, "--model", model_path, "--contextsize", "4096", "--port", "5001"]

    print("📟 Launching KoboldCpp...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        # Start KoboldCpp in background
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        print("⏳ Waiting for server to start...")

        # Check if server is ready
        server_ready = await check_server_ready()

        if server_ready:
            print("\n" + "=" * 70)
            print("🎯 Running feedback collection...")
            print("=" * 70 + "\n")

            # Run the feedback collection
            await run_feedback_collection()

            print("\n" + "=" * 70)
            print("✨ Test complete!")
            print("\nPress Enter to shutdown KoboldCpp...")
            input()

        else:
            print("❌ Server failed to start. Check the output above for errors.")

    except FileNotFoundError:
        print("❌ Could not find koboldcpp.exe")
        print("Please make sure koboldcpp.exe is in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup - terminate KoboldCpp
        if "process" in locals():
            print("\n🛑 Shutting down KoboldCpp...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("✅ KoboldCpp shut down")


if __name__ == "__main__":
    asyncio.run(main())
