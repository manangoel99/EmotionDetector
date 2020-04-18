document.getElementById("file-upload").style.visibility = "hidden";
document.getElementById('file-browser').

$("#file-browser").click(function(e) {
    e.preventDefault();
    $("#file-upload").trigger("click");
});