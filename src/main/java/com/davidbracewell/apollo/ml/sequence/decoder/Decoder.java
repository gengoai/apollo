package com.davidbracewell.apollo.ml.sequence.decoder;

import com.davidbracewell.apollo.ml.sequence.Labeling;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;

/**
 * @author David B. Bracewell
 */
public interface Decoder {

  Labeling decode(SequenceLabeler labeler, Sequence sequence);

}//END OF Decoder
