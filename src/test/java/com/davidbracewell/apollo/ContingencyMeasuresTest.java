package com.davidbracewell.apollo;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author David B. Bracewell
 */
public class ContingencyMeasuresTest {
  ContingencyTable table;

  @Before
  public void setUp() throws Exception {
    table = ContingencyTable.create2X2(
      20,
      42,
      27,
      14_000_000
    );
  }

  @Test
  public void testMI() throws Exception {
    assertEquals(2.52E-05, ContingencyMeasures.MI.calculate(table), 0.01);
  }

  @Test
  public void testX2() throws Exception {
    assertEquals(4938255.94, ContingencyMeasures.CHI_SQUARE.calculate(table), 0.01);
  }

  @Test
  public void testPMI() throws Exception {
    assertEquals(17.91, ContingencyMeasures.PMI.calculate(table), 0.01);
  }

  @Test
  public void testOdds() throws Exception {
    assertEquals(1818175.45, ContingencyMeasures.ODDS_RATIO.calculate(table), 0.01);
  }

  @Test
  public void testPS() throws Exception {
    assertEquals(228.34, ContingencyMeasures.POISSON_STIRLING.calculate(table), 0.01);
  }

  @Test
  public void testNPMI() throws Exception {
    assertEquals(0.923, ContingencyMeasures.NPMI.calculate(table), 0.01);
  }

  @Test
  public void testT() throws Exception {
    assertEquals(4.47, ContingencyMeasures.T_SCORE.calculate(table), 0.01);
  }

  @Test
  public void testLL() throws Exception {
    assertEquals(489.32, ContingencyMeasures.LOG_LIKELIHOOD.calculate(table), 0.01);
  }
}