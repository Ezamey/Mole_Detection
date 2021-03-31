$(function() {
    $('a#buttonLinkClose').bind('click', function(event) {
      event.preventDefault();
      $.getJSON('/close_feed',
          function(data) {
        //do nothing
      });
  return false;
});


  $('a#buttonLinkCapture').bind('click', function(event) {
    event.preventDefault();
    $.get('http://127.0.0.1:5000/capture_feed',
        function(data) {
          $('.myPred').attr("src",data);
          console.log('coucou')
          console.log(data)
    });
  return false;
  });
});