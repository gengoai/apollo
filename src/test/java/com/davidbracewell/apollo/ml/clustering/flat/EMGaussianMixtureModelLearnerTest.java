package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.ClustererTest;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class EMGaussianMixtureModelLearnerTest extends ClustererTest {

   public EMGaussianMixtureModelLearnerTest() {
      super(new EMGaussianMixtureModelLearner()
               .setParameter("k", 2));
   }

   @Test
   public void testCluster() throws Exception {
      Clustering c = cluster();
      assertEquals(2, c.size());
      double[] dist = c.softCluster(Instance.create(Arrays.asList(
         Feature.real("F1", 0.5),
         Feature.real("F2", 1.5))));
      assertTrue(dist[0] >= 0);
      assertTrue(dist[1] >= 0);
   }

}