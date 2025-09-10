"""
Caching System for Character Simulation Engine
Provides in-memory and optional Redis caching with TTL support
"""

import json
import hashlib
import time
import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl <= 0:
            return False  # No expiration
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

class CacheManager:
    """Base cache manager with in-memory storage"""
    
    def __init__(self, 
                 max_size_mb: float = 100,
                 default_ttl: int = 3600,
                 enable_stats: bool = True):
        """
        Initialize cache manager
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl: Default time-to-live in seconds
            enable_stats: Whether to track cache statistics
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        logger.info(f"Initialized CacheManager: max_size={max_size_mb}MB, ttl={default_ttl}s")
    
    def _generate_key(self, 
                     prompt: str, 
                     temperature: float,
                     model: Optional[str] = None) -> str:
        """Generate cache key from prompt and parameters"""
        # Create a deterministic key from inputs
        key_data = {
            "prompt": prompt,
            "temperature": round(temperature, 2),  # Round to avoid float precision issues
            "model": model or "default"
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, 
                  prompt: str,
                  temperature: float,
                  model: Optional[str] = None) -> Optional[Any]:
        """Retrieve value from cache"""
        async with self._lock:
            key = self._generate_key(prompt, temperature, model)
            
            if self.enable_stats:
                self.stats["total_requests"] += 1
            
            if key not in self.cache:
                if self.enable_stats:
                    self.stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key[:8]}...")
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                if self.enable_stats:
                    self.stats["misses"] += 1
                logger.debug(f"Cache expired for key: {key[:8]}...")
                return None
            
            # Update access stats
            entry.access()
            
            if self.enable_stats:
                self.stats["hits"] += 1
            
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return entry.value
    
    async def set(self,
                  prompt: str,
                  temperature: float,
                  value: Any,
                  model: Optional[str] = None,
                  ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        async with self._lock:
            key = self._generate_key(prompt, temperature, model)
            
            # Check cache size
            await self._check_size_limit()
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl or self.default_ttl
            )
            
            self.cache[key] = entry
            logger.debug(f"Cached value for key: {key[:8]}...")
            return True
    
    async def _check_size_limit(self):
        """Evict old entries if cache is too large"""
        # Estimate cache size (simplified)
        estimated_size = len(pickle.dumps(self.cache))
        
        if estimated_size > self.max_size_bytes:
            # Evict least recently used entries
            entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed or x[1].created_at
            )
            
            # Remove oldest 20% of entries
            evict_count = len(entries) // 5
            for key, _ in entries[:evict_count]:
                del self.cache[key]
                if self.enable_stats:
                    self.stats["evictions"] += 1
            
            logger.info(f"Evicted {evict_count} cache entries due to size limit")
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enable_stats:
            return {}
        
        total = self.stats["total_requests"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "estimated_bytes": len(pickle.dumps(self.cache))
        }
    
    async def cleanup_expired(self):
        """Remove expired entries"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

class SimulationCache(CacheManager):
    """Specialized cache for character simulations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_variations = {}  # Track similar prompts
    
    async def get_simulation(self,
                            character_id: str,
                            situation: str,
                            emphasis: str,
                            temperature: float) -> Optional[Dict]:
        """Get cached simulation result"""
        # Create composite key
        prompt = f"{character_id}|{situation}|{emphasis}"
        result = await self.get(prompt, temperature, model="simulation")
        
        if result:
            logger.info(f"Retrieved cached simulation for {character_id}/{emphasis}")
        
        return result
    
    async def set_simulation(self,
                            character_id: str,
                            situation: str,
                            emphasis: str,
                            temperature: float,
                            response: Dict,
                            ttl: Optional[int] = None) -> bool:
        """Cache simulation result"""
        prompt = f"{character_id}|{situation}|{emphasis}"
        success = await self.set(prompt, temperature, response, 
                                model="simulation", ttl=ttl)
        
        if success:
            logger.info(f"Cached simulation for {character_id}/{emphasis}")
        
        return success
    
    async def get_similar_simulations(self,
                                     character_id: str,
                                     situation: str,
                                     threshold: float = 0.8) -> list:
        """Find similar cached simulations"""
        similar = []
        
        async with self._lock:
            for key, entry in self.cache.items():
                if entry.is_expired():
                    continue
                    
                # Simple similarity check based on character
                if character_id in key:
                    similar.append({
                        "key": key,
                        "value": entry.value,
                        "created_at": entry.created_at
                    })
        
        return similar[:5]  # Return top 5 similar

class RedisCache(CacheManager):
    """Redis-backed cache implementation"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 **kwargs):
        """Initialize Redis cache"""
        super().__init__(**kwargs)
        
        # Note: In production, initialize redis client here
        # import redis.asyncio as redis
        # self.redis = redis.Redis(host=host, port=port, db=db, password=password)
        
        self.redis_config = {
            "host": host,
            "port": port,
            "db": db
        }
        logger.info(f"Redis cache configured: {host}:{port}/{db}")
    
    async def get(self, prompt: str, temperature: float, model: Optional[str] = None) -> Optional[Any]:
        """Get from Redis first, fallback to memory"""
        # In production: check Redis first
        # key = self._generate_key(prompt, temperature, model)
        # value = await self.redis.get(key)
        # if value:
        #     return pickle.loads(value)
        
        # Fallback to in-memory
        return await super().get(prompt, temperature, model)
    
    async def set(self, prompt: str, temperature: float, value: Any, 
                  model: Optional[str] = None, ttl: Optional[int] = None) -> bool:
        """Set in both Redis and memory"""
        # Store in memory
        success = await super().set(prompt, temperature, value, model, ttl)
        
        # In production: also store in Redis
        # if success:
        #     key = self._generate_key(prompt, temperature, model)
        #     await self.redis.setex(key, ttl or self.default_ttl, pickle.dumps(value))
        
        return success

# Factory function
def create_cache(config: Dict[str, Any]) -> CacheManager:
    """Create cache instance based on configuration"""
    
    cache_config = config.get("cache", {})
    
    if not cache_config.get("enabled", True):
        logger.info("Cache disabled in configuration")
        return None
    
    redis_config = cache_config.get("redis", {})
    
    if redis_config.get("enabled", False):
        logger.info("Creating Redis cache")
        return RedisCache(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            max_size_mb=cache_config.get("max_size_mb", 100),
            default_ttl=cache_config.get("ttl_seconds", 3600)
        )
    else:
        logger.info("Creating in-memory cache")
        return SimulationCache(
            max_size_mb=cache_config.get("max_size_mb", 100),
            default_ttl=cache_config.get("ttl_seconds", 3600)
        )

# Background cleanup task
async def cache_maintenance_task(cache: CacheManager, interval: int = 300):
    """Periodically clean up expired cache entries"""
    while True:
        await asyncio.sleep(interval)
        await cache.cleanup_expired()
        
        stats = cache.get_stats()
        if stats:
            logger.info(f"Cache stats: hit_rate={stats['hit_rate']:.2%}, size={stats['cache_size']}")

if __name__ == "__main__":
    # Demo usage
    async def demo():
        # Create cache
        cache = SimulationCache(max_size_mb=10, default_ttl=60)
        
        # Store simulation
        await cache.set_simulation(
            character_id="pilate",
            situation="Trial scene",
            emphasis="power",
            temperature=0.9,
            response={"dialogue": "The empire demands order!"}
        )
        
        # Retrieve simulation
        result = await cache.get_simulation(
            character_id="pilate",
            situation="Trial scene",
            emphasis="power",
            temperature=0.9
        )
        
        print(f"Retrieved: {result}")
        
        # Check stats
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
    
    asyncio.run(demo())