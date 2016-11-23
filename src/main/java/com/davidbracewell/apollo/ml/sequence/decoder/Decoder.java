package com.davidbracewell.apollo.ml.sequence.decoder;

import com.davidbracewell.apollo.ml.sequence.Labeling;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;

/**
 * <p>Labels a sequence using a given {@link SequenceLabeler}</p>
 *
 * @author David B. Bracewell
 */
public interface Decoder {

   /**
    * Label the given sequence using the given labeler.
    *
    * @param labeler  the labeler to use
    * @param sequence the sequence to label
    * @return the labeling
    */
   Labeling decode(SequenceLabeler labeler, Sequence sequence);

}//END OF Decoder
