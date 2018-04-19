package com.gengoai.apollo.stat.measure;

import com.gengoai.apollo.Optimum;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class SimilarityTest {

   final NDArray v1 = NDArrayFactory.wrap(new double[]{
      0.06017036811209875, 0.6200632004644884, 0.7609311190726616, 0.49296225829243057, 0.4550247106358657, 0.46427358814876685, 0.19906369044188432, 0.6462803017542045, 0.5371970147563676, 0.4244205654374822});
   final NDArray v2 = NDArrayFactory.wrap(
      new double[]{0.07114390080558797, 0.246121098350624, 0.778677300972233, 0.2843022452280499, 0.9286238183735879, 0.9800765792531674, 0.595180121663983, 0.42473313393664824, 0.8406326336670255, 0.31775147612780663});

   final ContingencyTable table = ContingencyTable.create2X2(10, 20, 25, 45);

   @Test
   public void angular() {
      assertEquals(0.8342426651585743, Similarity.Angular.calculate(v1, v2), 0.0001);
   }

   @Test
   public void cosine() {
      assertEquals(0.8674502431089217, Similarity.Cosine.calculate(v1, v2), 0.0001);
      assertEquals(1d - 0.8674502431089217, Similarity.Cosine.asDistanceMeasure().calculate(v1, v2), 0.0001);
      assertEquals(0.8674502431089217, Similarity.Cosine.calculate(v1.toArray(), v2.toArray()), 0.0001);
      assertEquals(0.3108496602871197, Similarity.Cosine.calculate(table), 0.0001);
      Assert.assertEquals(Optimum.MAXIMUM, Similarity.Cosine.getOptimum());
   }

   @Test
   public void dice() {
      assertEquals(0.7404632216185192, Similarity.Dice.calculate(v1, v2), 0.0001);
      assertEquals(0.4444444444444444, Similarity.Dice.calculate(table), 0.0001);
   }

   @Test
   public void dicegen2() {
      assertEquals(0.27119406623002607, Similarity.DiceGen2.calculate(v1, v2), 0.0001);
      assertEquals(0.4444444444444444, Similarity.DiceGen2.calculate(table), 0.0001);
   }

   @Test
   public void dot() {
      assertEquals(2.7465529238126223, Similarity.DotProduct.calculate(v1, v2), 0.0001);
      assertEquals(10.0, Similarity.DotProduct.calculate(table), 0.0001);
   }

   @Test
   public void jaccard() {
      assertEquals(0.5878853514464445, Similarity.Jaccard.calculate(v1, v2), 0.0001);
      assertEquals(0.2857142857142857, Similarity.Jaccard.calculate(table), 0.0001);
   }

   @Test
   public void overlap() {
      assertEquals(1.0, Similarity.Overlap.calculate(v1, v2), 0.0001);
      assertEquals(0.5, Similarity.Overlap.calculate(table), 0.0001);
   }

}