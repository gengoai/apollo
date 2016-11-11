package com.davidbracewell.apollo.ml;

import com.davidbracewell.cache.CacheManager;
import com.davidbracewell.cache.Cached;
import com.davidbracewell.cache.KeyMaker;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.config.Config;
import com.davidbracewell.conversion.Cast;
import com.google.common.collect.Iterables;
import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Method;
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


   public static class TestKeyMaker implements KeyMaker {
      private static final long serialVersionUID = 1L;

      @Override
      public Object make(Class<?> clazz, Method method, Object[] args) {
         return clazz.getName() + "::" + method.getName() + "::" + args.length;
      }
   }

   @Cached(keyMaker = TestKeyMaker.class)
   private static class SimpleBinaryFeaturizer extends BinaryFeaturizer<String> {
      private static final long serialVersionUID = 1L;

      protected Set<String> applyImpl(String input) {
         return Collections.singleton(input);
      }
   }

   @Cached(keyMaker = TestKeyMaker.class, name = CacheManager.GLOBAL_CACHE)
   private static class SimpleRealFeaturizer extends RealFeaturizer<String> {
      private static final long serialVersionUID = 1L;

      protected Counter<String> applyImpl(String input) {
         return Counters.newCounter(input);
      }
   }

   @Test
   public void normalFeaturizer() throws Exception {
      assertEquals(Collections.singleton(Feature.TRUE("test")), f1.apply("test"));
   }

   @Test
   public void cachedBinary() throws Exception {
      f2.apply("test");
      assertEquals(Feature.TRUE("test"), getFeature(SimpleBinaryFeaturizer.class.getName()));
   }

   @Test
   public void cachedReal() throws Exception {
      f3.apply("test");
      assertEquals(Feature.TRUE("test"), getFeature(SimpleRealFeaturizer.class.getName()));
   }

   @Before
   public void setUp() throws Exception {
      Config.initializeTest();
      f1 = ((Featurizer<String>) s -> Collections.singleton(Feature.TRUE(s))).cache(CacheManager.GLOBAL_CACHE);
      f2 = new SimpleBinaryFeaturizer().cache(CacheManager.GLOBAL_CACHE);
      f3 = new SimpleRealFeaturizer().cache();
   }

   private Feature getFeature(String className) {
      String f2CacheKey = className + "::apply::1";
      Object o = CacheManager.getGlobalCache().get(f2CacheKey);
      if (o == null) {
         return null;
      }
      Set<Feature> cached = Cast.as(o);
      return Iterables.getFirst(cached, null);
   }


}// END OF FeatureCacheTest
