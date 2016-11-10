package com.davidbracewell.apollo.ml;

import com.davidbracewell.cache.CacheManager;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.config.Config;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class FeatureCacheTest {

   Featurizer<String> f1;
   Featurizer<String> f2;
   Featurizer<String> f3;

   @Test
   public void normalFeaturizer() throws Exception {
      assertEquals(Collections.singleton(Feature.TRUE("test")), f1.apply("test"));
      System.out.println(CacheManager.get(CacheManager.GLOBAL_CACHE).size());
   }

   @Before
   public void setUp() throws Exception {
      Config.initializeTest();
      f1 = ((Featurizer<String>) s -> Collections.singleton(Feature.TRUE(s))).cache(CacheManager.GLOBAL_CACHE);
      f2 = new BinaryFeaturizer<String>() {
         @Override
         protected Set<String> applyImpl(String input) {
            return Collections.singleton(input);
         }
      }.cache(CacheManager.GLOBAL_CACHE);
      f3 = new RealFeaturizer<String>() {
         @Override
         protected Counter<String> applyImpl(String input) {
            return Counters.newCounter(input);
         }
      }.cache(CacheManager.GLOBAL_CACHE);
   }
}// END OF FeatureCacheTest
