#!/usr/bin/env python3
"""
Comprehensive test suite for the HackRX Policy Analysis API
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000", api_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token or os.getenv("API_TOKEN")
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    async def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        logger.info("Testing health check endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Health check passed: {data.get('status')}")
                    return True
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        logger.info("Testing root endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Root endpoint response: {data.get('message')}")
                    return True
                else:
                    logger.error(f"Root endpoint failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Root endpoint error: {e}")
            return False
    
    async def test_authentication(self) -> bool:
        """Test authentication with invalid token."""
        logger.info("Testing authentication...")
        try:
            headers = {"Authorization": "Bearer invalid_token", "Content-Type": "application/json"}
            test_payload = {
                "documents": "https://example.com/test.pdf",
                "questions": ["Test question?"]
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=headers,
                json=test_payload
            ) as response:
                if response.status == 401:
                    logger.info("Authentication test passed - invalid token rejected")
                    return True
                else:
                    logger.error(f"Authentication test failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Authentication test error: {e}")
            return False
    
    async def test_query_processing(self, test_document_url: str, test_questions: List[str]) -> Dict[str, Any]:
        """Test the main query processing endpoint."""
        logger.info("Testing query processing...")
        
        test_payload = {
            "documents": test_document_url,
            "questions": test_questions
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self._get_headers(),
                json=test_payload
            ) as response:
                end_time = time.time()
                processing_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    result = {
                        "success": True,
                        "status_code": response.status,
                        "processing_time": processing_time,
                        "answers_count": len(data.get("answers", [])),
                        "has_metadata": "metadata" in data,
                        "response_data": data
                    }
                    
                    logger.info(f"Query processing test passed in {processing_time:.2f}s")
                    logger.info(f"Received {result['answers_count']} answers")
                    
                    return result
                else:
                    error_text = await response.text()
                    result = {
                        "success": False,
                        "status_code": response.status,
                        "processing_time": processing_time,
                        "error": error_text
                    }
                    logger.error(f"Query processing test failed: {response.status} - {error_text}")
                    return result
                    
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            logger.error(f"Query processing test error: {e}")
            return result
    
    async def test_invalid_document_url(self) -> bool:
        """Test with invalid document URL."""
        logger.info("Testing invalid document URL...")
        
        test_payload = {
            "documents": "https://invalid-url-that-does-not-exist.com/fake.pdf",
            "questions": ["Test question?"]
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self._get_headers(),
                json=test_payload
            ) as response:
                if response.status >= 400:
                    logger.info("Invalid document URL test passed - error returned as expected")
                    return True
                else:
                    logger.error(f"Invalid document URL test failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Invalid document URL test error: {e}")
            return False
    
    async def test_empty_questions(self) -> bool:
        """Test with empty questions list."""
        logger.info("Testing empty questions...")
        
        test_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": []
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self._get_headers(),
                json=test_payload
            ) as response:
                data = await response.json()
                if response.status == 200 and len(data.get("answers", [])) == 0:
                    logger.info("Empty questions test passed")
                    return True
                else:
                    logger.error(f"Empty questions test failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Empty questions test error: {e}")
            return False
    
    async def test_analytics_endpoint(self) -> bool:
        """Test the analytics endpoint."""
        logger.info("Testing analytics endpoint...")
        try:
            async with self.session.get(
                f"{self.base_url}/api/v1/analytics",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Analytics endpoint passed: {data.get('total_queries', 0)} total queries")
                    return True
                else:
                    logger.warning(f"Analytics endpoint returned: {response.status}")
                    return response.status in [503]  # Service unavailable is acceptable
        except Exception as e:
            logger.error(f"Analytics endpoint error: {e}")
            return False
    
    async def run_comprehensive_tests(self):
        """Run all tests in sequence."""
        logger.info("Starting comprehensive API tests...")
        
        # Test document URL (using the provided example)
        test_document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
        # Test questions from the example
        test_questions = [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
        
        tests = [
            ("Health Check", self.test_health_check()),
            ("Root Endpoint", self.test_root_endpoint()),
            ("Authentication", self.test_authentication()),
            ("Query Processing", self.test_query_processing(test_document_url, test_questions)),
            ("Invalid Document URL", self.test_invalid_document_url()),
            ("Empty Questions", self.test_empty_questions()),
            ("Analytics Endpoint", self.test_analytics_endpoint())
        ]
        
        results = {}
        
        for test_name, test_coro in tests:
            try:
                logger.info(f"\n--- Running {test_name} Test ---")
                result = await test_coro
                results[test_name] = result
                
                if isinstance(result, bool):
                    status = "PASSED" if result else "FAILED"
                    logger.info(f"{test_name}: {status}")
                elif isinstance(result, dict):
                    status = "PASSED" if result.get("success", False) else "FAILED"
                    logger.info(f"{test_name}: {status}")
                    if not result.get("success", False):
                        logger.error(f"Error details: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"{test_name} test failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n--- Test Summary ---")
        total_tests = len(tests)
        passed_tests = 0
        
        for test_name, result in results.items():
            if isinstance(result, bool):
                passed = result
            elif isinstance(result, dict):
                passed = result.get("success", False)
            else:
                passed = False
            
            if passed:
                passed_tests += 1
            
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if "Query Processing" in results and isinstance(results["Query Processing"], dict):
            query_result = results["Query Processing"]
            if query_result.get("success"):
                logger.info(f"Query processing time: {query_result.get('processing_time', 0):.2f}s")
                logger.info(f"Answers received: {query_result.get('answers_count', 0)}")
        
        return results

async def main():
    """Main test runner."""
    # Configuration
    base_url = os.getenv("TEST_BASE_URL", "http://localhost:8000")
    api_token = os.getenv("API_TOKEN")
    
    if not api_token:
        logger.error("API_TOKEN not found in environment variables")
        return
    
    logger.info(f"Testing API at: {base_url}")
    logger.info(f"Using API token: {api_token[:10]}..." if api_token else "No token")
    
    async with APITester(base_url, api_token) as tester:
        results = await tester.run_comprehensive_tests()
        
        # Generate test report
        report_file = "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nTest report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
