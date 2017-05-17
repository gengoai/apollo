package com.davidbracewell.apollo.linalg;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface MatrixFactory extends Serializable {

    Matrix create(int numRows, int numColumns);

}// END OF MatrixFactory
