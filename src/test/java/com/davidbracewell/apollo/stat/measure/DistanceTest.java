package com.davidbracewell.apollo.stat.measure;

import com.davidbracewell.apollo.Optimum;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class DistanceTest {

   final NDArray v1 = NDArrayFactory.wrap(new double[]{
      0.06017036811209875, 0.6200632004644884, 0.7609311190726616, 0.49296225829243057, 0.4550247106358657, 0.46427358814876685, 0.19906369044188432, 0.6462803017542045, 0.5371970147563676, 0.4244205654374822});
   final NDArray v2 = NDArrayFactory.wrap(
      new double[]{0.07114390080558797, 0.246121098350624, 0.778677300972233, 0.2843022452280499, 0.9286238183735879, 0.9800765792531674, 0.595180121663983, 0.42473313393664824, 0.8406326336670255, 0.31775147612780663});

   @Test
   public void Chebyshev() {
      assertEquals(0.5158029911044005, Distance.Chebyshev.calculate(v1, v2), 0.0001);
   }

   @Test
   public void EarthMovers() {
      assertEquals(4.950955044299373, Distance.EarthMovers.calculate(v1, v2), 0.0001);
   }

   @Test
   public void Euclidean() {
      assertEquals(0.9917654595464219, Distance.Euclidean.calculate(v1, v2), 0.0001);
      assertEquals(-0.9917654595464219, Distance.Euclidean.asSimilarityMeasure().calculate(v1, v2), 0.0001);
      assertEquals(Optimum.MINIMUM, Distance.Euclidean.getOptimum());
   }

   @Test
   public void Hamming() {
      assertEquals(10, Distance.Hamming.calculate(v1, v2), 0.0001);
   }

   @Test
   public void KLDivergence() {
      assertEquals(0.0807280628357561, Distance.KLDivergence.calculate(v1, v2), 0.0001);
   }

   @Test
   public void Manhattan() {
      assertEquals(2.6284922358734164, Distance.Manhattan.calculate(v1, v2), 0.0001);
   }

   @Test
   public void SquaredEuclidean() {
      assertEquals(0.9835987267493254, Distance.SquaredEuclidean.calculate(v1, v2), 0.0001);
   }

}